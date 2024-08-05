import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import cv2
from .utils import *

from .asr import ASR
##Hojun added
from queue import Queue
import threading
import subprocess

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([1, 0, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def update_pose(self, pose):
        # pose: [4, 4] numpy array
        # assert self.center is 0
        self.radius = np.linalg.norm(pose[:3, 3])
        T = np.eye(4)
        T[2, 3] = -self.radius
        rot = pose @ np.linalg.inv(T)
        self.rot = R.from_matrix(rot[:3, :3])

    def update_intrinsics(self, intrinsics):
        fl_x, fl_y, cx, cy = intrinsics
        self.W = int(cx * 2)
        self.H = int(cy * 2)
        self.fovy = np.rad2deg(2 * np.arctan2(self.H, 2 * fl_y))  ##rad2deg Convert angles from radians to degrees
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.deg2rad(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.01 * dx)
        rotvec_y = side * np.radians(-0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(self, opt, trainer, data_loader, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        print('aaaaaaaaaaaa',self.W)
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.data_loader = data_loader
        # print('dataloader length: ', len(self.data_loader))

        # override with dataloader's intrinsics
        self.W = data_loader._data.W
        print('bbbbbbbbbbbbb', self.W)
        self.H = data_loader._data.H
        self.cam.update_intrinsics(data_loader._data.intrinsics)

        # use dataloader's pose
        pose_init = data_loader._data.poses[0]   
        self.cam.update_pose(pose_init.detach().cpu().numpy())

        # use dataloader's bg
        bg_img = data_loader._data.bg_img #.view(1, -1, 3)
        if self.H != bg_img.shape[0] or self.W != bg_img.shape[1]:
            bg_img = F.interpolate(bg_img.permute(2, 0, 1).unsqueeze(0).contiguous(), (self.H, self.W), mode='bilinear').squeeze(0).permute(1, 2, 0).contiguous()
        self.bg_color = bg_img.view(1, -1, 3)

        # audio features (from dataloader, only used in non-playing mode)  ##실시간 모드할때는 사용안됨 
        self.audio_features = data_loader._data.auds # [N, 29, 16]   ##--asr 이 인자로 안되어있으면 None이 return된다
        self.audio_idx = 0

        # control eye
        self.eye_area = None if not self.opt.exp_eye else data_loader._data.eye_area.mean().item()

        # playing seq from dataloader, or pause.
        # self.playing = opt.playing
        # self.playing = False
        self.playing = True
        self.loader = iter(data_loader)

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = False # assert False!
        self.downscale = 1
        self.train_steps = 16

        self.ind_index = 0
        self.ind_num = trainer.model.individual_codes.shape[0]

        # build asr
        if self.opt.asr:
            self.asr = ASR(opt)
   
        # dpg.create_context()
        # self.register_dpg()
        # self.test_step()
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opt.asr:
            self.asr.stop()        
        # dpg.destroy_context()

    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.data_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        # dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        # dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs): ##결과물을 image로 뱉어낼것이냐 아니면 depth이밎로 아웃풋할 것이냐 결정하는 부분
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)  

    def test_step(self):
        
        
        # if self.need_update or self.spp < self.opt.max_spp:
        
        #     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #     starter.record()

            if self.playing:  ##즉, args.playing True로 해야지 real time inferencing을 한다는뜻?
        
                q = Queue()
                # for i in range(100):
                #     try:
                #         data = next(self.loader)
                #         # print('aaaaaaaaaaaaaaaaaaaaaaa', len(self.loader))
                #     except StopIteration:
                #         self.loader = iter(self.data_loader)
                #         ## len(self.loader) = 14544
                #         data = next(self.loader)
                    
                #     if self.opt.asr:
                #         # use the live audio stream
                #         data['auds'] = self.asr.get_next_feat()
                # for data in self.loader:
                #     if self.opt.asr:
                #         data['auds'] = self.asr.get_next_feat()
        
                inference_thread  = threading.Thread(target=self.trainer.test_gui_with_data, args=(self.loader, self.W, self.H, q))
                inference_thread.start()
                # outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
                print('this is q data', q)
                print('ququququeueue', q.queue)
            
                # # FFmpeg에 파이프하는 명령 설정 (오디오 포함)
                # command = ['ffmpeg',
                #         '-y',  # 기존 파일 덮어쓰기
                #         '-i',  '-', ##video frame은 pipe로 입력받기
                #         '-f', 'libx264',
                #         '-vcodec', 'libx264',  ##https://stackoverflow.com/questions/7238013/rawvideo-and-rgb32-values-passed-to-ffmpeg
                #         '-pix_fmt', 'bgr24',
                #         '-c:v', 'libx264',
                #         '-s', '512x512',  # 크기 설정
                #         '-r', '25',  # 프레임 속도
                #         '-i', 'data/obama/trump_28.wav',  # 오디오 파일 추가 ##-i = input 
                #         '-c:a', 'aac',  # 오디오 코덱 설정
                #         '-strict', 'experimental',
                #         '-pix_fmt', 'yuv420p',
                #         '-preset', 'ultrafast',
                #         '-g', '20',
                #         '-hls_time', '5',  # 각 세그먼트의 길이를 10초로 설정
                #         '-hls_list_size', '0',  # 무한 재생 목록 크기
                #         '-f', 'hls',  # HLS 포맷 사용
                #         'video/output.m3u8']  # 출력 파일 이름
            
                #         '-i', '-',  # 파이프로 입력 받기
                #         '-i', 'data/obama/trump_28.wav',  # 오디오 파일 추가
                #         '-c:v', 'libx264',
                #         '-c:a', 'aac',  # 오디오 코덱 설정
                #         '-strict', 'experimental',
                #         '-pix_fmt', 'yuv420p',
                #         '-preset', 'ultrafast',
                #         '-g', '20',
                #         '-hls_time', '2',  # 각 세그먼트의 길이를 10초로 설정
                #         '-hls_list_size', '0',  # 무한 재생 목록 크기
                #         '-f', 'hls',  # HLS 포맷 사용
                #         'video/output.m3u8']  # 출력 파일 이름
            
                # command = ['ffmpeg',
                #         '-y',  # 기존 파일 덮어쓰기
                #         '-f', 'rawvideo',
                #         '-vcodec', 'rawvideo',
                #         '-pix_fmt', 'bgr24',
                #         '-s', '512x512',  # 크기 설정
                #         '-r', '25',  # 프레임 속도
                #         '-i', '-',  # 파이프로 입력 받기
                #         '-i', 'data/obama/trump_28.wav',  # 오디오 파일 추가
                #         '-c:v', 'libx264',  ##코덱인데 오디오라는 의미 ( -c:v 이면 비디오라는 뜻 ) 
                #         '-c:a', 'aac',  # 오디오 코덱 설정  ##코덱인데 오디오라는 의미 ( -c:v 이면 비디오라는 뜻 ) 
                #         '-strict', 'experimental',  ##https://m.blog.naver.com/josm17/220601051707
                #         '-pix_fmt', 'yuv420p',  ##pixel format selection
                #         '-preset', 'ultrafast',
                #         '-g', '20',
                #         '-hls_time', '2',  # 각 세그먼트의 길이를 10초로 설정
                #         '-hls_list_size', '0',  # 무한 재생 목록 크기
                #         '-f', 'hls',  # HLS 포맷 사용
                #         'video/justin_incheon.m3u8']  # 출력 파일 이름
                
                command = ['ffmpeg',
                           '-y',  # 기존 파일 덮어쓰기
                           '-f', 'rawvideo',
                        #    '-vsync', '2',
                        #    '-thread_queue_size', '5000',
                           '-vcodec', 'rawvideo',
                           '-pix_fmt', 'bgr24',
                           '-s', '512x512',  # 크기 설정
                           '-r', '25',  # 프레임 속도
                           '-thread_queue_size', '64',
                           '-i', '-',  # 파이프로 입력 받기
                           '-thread_queue_size', '64',
                           '-i', 'data/obama/trump_28.wav',  # 오디오 파일 추가
                           '-c:v', 'libx264',
                           '-c:a', 'aac',  # 오디오 코덱 설정
                           '-strict', 'experimental',
                           '-pix_fmt', 'yuv420p',
                        #    '-preset', 'ultrafast',
                           '-preset', 'fast',
                           '-g', '20',
                           '-hls_time', '2',  # 각 세그먼트의 길이를 10초로 설정
                           '-hls_list_size', '0',  # 무한 재생 목록 크기
                           '-f', 'hls', 
                           # HLS 포맷 사용
                           'video/output.m3u8']  # 출력 파일 이름

                # FFmpeg 프로세스 시작
                p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   ##https://blog.naver.com/PostView.naver?blogId=sagala_soske&logNo=222131573917
                
                # 이미지를 회색조로 처리하고 FFmpeg에 전송
                frame_rate = 1.0 / 30  # 25 FPS에 해당하는 시간 간격
                import time
                cnt = 0
                
                while True:
                    # if q.qsize() <= 4:
                    #     continue
        
                
                # while True:
                    # # time.sleep(10)
                    # print('qsize', q.qsize())
                
                    if q:
                        cnt += 1
                        outputs = q.get()  ##outputs = image frame(s)
                        # outputs = F.interpolate(outputs.permute(0, 3, 1, 2), size=(512, 512), mode='bilinear').permute(0, 2, 3, 1).contiguous()
                        # outputs = outputs.detach().cpu().numpy()

                        # print(outputs)

                        
                        # print('justin must come to Incheon', outputs.shape)
                        # outputs = outputs.squeeze(0)
                        # print('squeezed incheon justin', outputs.shape)
                        # # print('outputs type', type(outputs))
                        # outputs = (outputs*255).astype(np.uint8)
                        # print('shapeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', outputs.shape)
                        # # print('this is denormalized queue get data', outputs)
                        # outputs = cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB)
                        
                        # cv2.imwrite('test_justin_incheon.png', outputs)
                        # print('dtype: ', outputs.dtype)
                        # print('tstype: ', type(outputs))

                        
                        # p.stdin.write(outputs.tobytes())  ##https://stackoverflow.com/questions/69188430/python-opencv-subprocess-write-return-broken-pipe
                        # p.stdin.write(outputs)
                        out, err = p.communicate(outputs.tobytes(), timeout=5)
                        print(out)
                        # print(err)
                        print(cnt)

                    elif p.returncode:
                        break                        

                    # elif cnt >= 15:
                    #     break

                p.stdin.close()
                p.wait()
                print(f"Succefully generated")
            
                        # print('check final value', outputs)
                        # outputs = cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB)
                        # p.communicate(outputs)##communicate = 프로세스와 상호 작용합니다: stdin에 데이터를 보냅니다.
                        # print('donezzzz')
                        # p.communicate(outputs.tobytes()) ##communicate = 프로세스와 상호 작용합니다: stdin에 데이터를 보냅니다.
                        # time.sleep(frame_rate)  # FPS에 맞춰 대기
                        
                    # else:
                    #     break

                    # import sys
                    # print(sys.argv)
                    # print('aa')
                # p.kill()
                # p.terminate()
                    
                # 완료되면 FFmpeg 닫기


                # sync local camera pose
                # self.cam.update_pose(data['poses_matrix'][0].detach().cpu().numpy())
        
            # else:
            #     if self.audio_features is not None:
            #         auds = get_audio_features(self.audio_features, self.opt.att, self.audio_idx)
            #     else:
            #         auds = None
            #     outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, auds, self.eye_area, self.ind_index, self.bg_color, self.spp, self.downscale)

            # ender.record()
            # torch.cuda.synchronize()
            # t = starter.elapsed_time(ender)
            


            # # update dynamic resolution
            # if self.dynamic_resolution:
            #     # max allowed infer time per-frame is 200 ms
            #     full_t = t / (self.downscale ** 2)
            #     downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
            #     if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
            #         self.downscale = downscale

            # if self.need_update:
            #     self.render_buffer = self.prepare_buffer(outputs)
            #     self.spp = 1
            #     self.need_update = False
            # else:
            #     self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
            #     self.spp += 1
            
            # if self.playing:
            #     self.need_update = True

            # dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            # dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            # dpg.set_value("_log_spp", self.spp)
            # dpg.set_value("_texture", self.render_buffer)

        
    # def register_dpg(self):

    #     ### register texture 

    #     with dpg.texture_registry(show=False):
    #         dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

    #     ### register window

    #     # the rendered image, as the primary window
    #     with dpg.window(tag="_primary_window", width=self.W, height=self.H):

    #         # add the texture
    #         dpg.add_image("_texture")

    #     # dpg.set_primary_window("_primary_window", True)
        
    #     dpg.show_tool(dpg.mvTool_Metrics)

    #     # control window
    #     with dpg.window(label="Control", tag="_control_window", width=400, height=300):

    #         # button theme
    #         with dpg.theme() as theme_button:
    #             with dpg.theme_component(dpg.mvButton):
    #                 dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
    #                 dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
    #                 dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
    #                 dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
    #                 dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

    #         # time
    #         if not self.opt.test:
    #             with dpg.group(horizontal=True):
    #                 dpg.add_text("Train time: ")
    #                 dpg.add_text("no data", tag="_log_train_time")                    ##필요한정보

    #         with dpg.group(horizontal=True):    ##필요한정보
    #             dpg.add_text("Infer time: ")    ##필요한정보
    #             dpg.add_text("no data", tag="_log_infer_time")  ##필요한정보
            
    #         with dpg.group(horizontal=True):
    #             dpg.add_text("SPP: ")
    #             dpg.add_text("1", tag="_log_spp")

    #         # train button
    #         if not self.opt.test:
    #             with dpg.collapsing_header(label="Train", default_open=True):

    #                 # train / stop
    #                 with dpg.group(horizontal=True):
    #                     dpg.add_text("Train: ")

    #                     def callback_train(sender, app_data):
    #                         if self.training:
    #                             self.training = False
    #                             dpg.configure_item("_button_train", label="start")
    #                         else:
    #                             self.training = True
    #                             dpg.configure_item("_button_train", label="stop")

    #                     dpg.add_button(label="start", tag="_button_train", callback=callback_train)
    #                     dpg.bind_item_theme("_button_train", theme_button)

    #                     def callback_reset(sender, app_data):
    #                         @torch.no_grad()
    #                         def weight_reset(m: nn.Module):
    #                             reset_parameters = getattr(m, "reset_parameters", None)
    #                             if callable(reset_parameters):
    #                                 m.reset_parameters()
    #                         self.trainer.model.apply(fn=weight_reset)
    #                         self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
    #                         self.need_update = True

    #                     dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
    #                     dpg.bind_item_theme("_button_reset", theme_button)

    #                 # save ckpt
    #                 with dpg.group(horizontal=True):
    #                     dpg.add_text("Checkpoint: ")

    #                     def callback_save(sender, app_data):
    #                         self.trainer.save_checkpoint(full=True, best=False)
    #                         dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
    #                         self.trainer.epoch += 1 # use epoch to indicate different calls.

    #                     dpg.add_button(label="save", tag="_button_save", callback=callback_save)
    #                     dpg.bind_item_theme("_button_save", theme_button)

    #                     dpg.add_text("", tag="_log_ckpt")
                    
    #                 # save meshplaying
    #                 with dpg.group(horizontal=True):
    #                     dpg.add_text("Marching Cubes: ")

    #                     def callback_mesh(sender, app_data):
    #                         self.trainer.save_mesh(resolution=256, threshold=10)
    #                         dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
    #                         self.trainer.epoch += 1 # use epoch to indicate different calls.

    #                     dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
    #                     dpg.bind_item_theme("_button_mesh", theme_button)

    #                     dpg.add_text("", tag="_log_mesh")

    #                 with dpg.group(horizontal=True):
    #                     dpg.add_text("", tag="_log_train_log")

            
    #         # rendering options
    #         with dpg.collapsing_header(label="Options", default_open=True):
                
    #             # playing
    #             with dpg.group(horizontal=True):
    #                 dpg.add_text("Play: ")

    #                 def callback_play(sender, app_data):
                        
    #                     if self.playing:
    #                         self.playing = False
    #                         dpg.configure_item("_button_play", label="start")
    #                     else:
    #                         self.playing = True
    #                         dpg.configure_item("_button_play", label="stop")
    #                         if self.opt.asr:
    #                             self.asr.warm_up()
    #                     self.need_update = True

    #                 dpg.add_button(label="start", tag="_button_play", callback=callback_play)
    #                 dpg.bind_item_theme("_button_play", theme_button)

    #                 # set asr
    #                 if self.opt.asr:

    #                     # clear queue button
    #                     def callback_clear_queue(sender, app_data):
                            
    #                         self.asr.clear_queue()
    #                         self.need_update = True

    #                     dpg.add_button(label="clear", tag="_button_clear_queue", callback=callback_clear_queue)
    #                     dpg.bind_item_theme("_button_clear_queue", theme_button)

    #             # dynamic rendering resolution
    #             with dpg.group(horizontal=True):
                    
    #                 def callback_set_dynamic_resolution(sender, app_data):

    #                     if self.dynamic_resolution:
    #                         self.dynamic_resolution = False
    #                         self.downscale = 1
    #                     else:
    #                         self.dynamic_resolution = True
    #                     self.need_update = True

    #                 # Disable dynamic resolution for face.
    #                 # dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
        
    #                 dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

    #             # mode combo
    #             def callback_change_mode(sender, app_data):
    #                 self.mode = app_data
    #                 self.need_update = True
                
    #             dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)


    #             # bg_color picker
    #             def callback_change_bg(sender, app_data):
    #                 self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
    #                 self.need_update = True

    #             dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

    #             # audio index slider
    #             if not self.opt.asr:
    #                 def callback_set_audio_index(sender, app_data):
    #                     self.audio_idx = app_data
    #                     self.need_update = True

    #                 dpg.add_slider_int(label="Audio", min_value=0, max_value=self.audio_features.shape[0] - 1, format="%d", default_value=self.audio_idx, callback=callback_set_audio_index)

    #             # ind code index slider
    #             if self.opt.ind_dim > 0:
    #                 def callback_set_individual_code(sender, app_data):
    #                     self.ind_index = app_data
    #                     self.need_update = True

    #                 dpg.add_slider_int(label="Individual", min_value=0, max_value=self.ind_num - 1, format="%d", default_value=self.ind_index, callback=callback_set_individual_code)

    #             # eye area slider
    #             if self.opt.exp_eye:
    #                 def callback_set_eye(sender, app_data):
    #                     self.eye_area = app_data
    #                     self.need_update = True

    #                 dpg.add_slider_float(label="eye area", min_value=0, max_value=0.5, format="%.2f percent", default_value=self.eye_area, callback=callback_set_eye)

    #             # fov slider
    #             def callback_set_fovy(sender, app_data):
    #                 self.cam.fovy = app_data
    #                 self.need_update = True

    #             dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

    #             # dt_gamma slider
    #             def callback_set_dt_gamma(sender, app_data):
    #                 self.opt.dt_gamma = app_data
    #                 self.need_update = True

    #             dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

    #             # max_steps slider
    #             def callback_set_max_steps(sender, app_data):
    #                 self.opt.max_steps = app_data
    #                 self.need_update = True

    #             dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

    #             # aabb slider
    #             def callback_set_aabb(sender, app_data, user_data):
    #                 # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
    #                 self.trainer.model.aabb_infer[user_data] = app_data

    #                 # also change train aabb ? [better not...]
    #                 #self.trainer.model.aabb_train[user_data] = app_data

    #                 self.need_update = True

    #             dpg.add_separator()
    #             dpg.add_text("Axis-aligned bounding box:")

    #             with dpg.group(horizontal=True):
    #                 dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
    #                 dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

    #             with dpg.group(horizontal=True):
    #                 dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
    #                 dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

    #             with dpg.group(horizontal=True):
    #                 dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
    #                 dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
                

    #         # debug info
    #         if self.debug:
    #             with dpg.collapsing_header(label="Debug"):
    #                 # pose
    #                 dpg.add_separator()
    #                 dpg.add_text("Camera Pose:")
    #                 dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        # def callback_camera_drag_rotate(sender, app_data):

        #     if not dpg.is_item_focused("_primary_window"):
        #         return

        #     dx = app_data[1]
        #     dy = app_data[2]

        #     self.cam.orbit(dx, dy)
        #     self.need_update = True

        #     if self.debug:
        #         dpg.set_value("_log_pose", str(self.cam.pose))


        # def callback_camera_wheel_scale(sender, app_data):

        #     if not dpg.is_item_focused("_primary_window"):
        #         return

        #     delta = app_data

        #     self.cam.scale(delta)
        #     self.need_update = True

        #     if self.debug:
        #         dpg.set_value("_log_pose", str(self.cam.pose))


        # def callback_camera_drag_pan(sender, app_data):

        #     if not dpg.is_item_focused("_primary_window"):
        #         return

        #     dx = app_data[1]
        #     dy = app_data[2]

        #     self.cam.pan(dx, dy)
        #     self.need_update = True

        #     if self.debug:
        #         dpg.set_value("_log_pose", str(self.cam.pose))


        # with dpg.handler_registry():
        #     dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
        #     dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
        #     dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        # dpg.create_viewport(title='RAD-NeRF', width=1080, height=720, resizable=True)

        # ### global theme
        # with dpg.theme() as theme_no_padding:
        #     with dpg.theme_component(dpg.mvAll):
        #         # set all padding to 0 to avoid scroll bar
        #         dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
        #         dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
        #         dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        # dpg.bind_item_theme("_primary_window", theme_no_padding)

        # dpg.setup_dearpygui()

        # #dpg.show_metrics()

        # dpg.show_viewport()


    def render(self):
    #    while dpg.is_dearpygui_running():
        # while True:            
        # update texture every frame
        if self.training:
            self.train_step()
        # audio stream thread...
        # if self.opt.asr and self.playing:
        #     # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
        #     for _ in range(2):
        #         self.asr.run_step()
        self.test_step()

            # dpg.render_dearpygui_frame()