# end-to-end test with audio streaming
python test.py \
    --pose data/obama/transforms_train.json \
    --ckpt trial_obama_torso/checkpoints/ngp_ep0028.pth \
    --aud data/obama/aud_eo.npy \
    --workspace trial_obama_torso \
    --bg_img data/obama/bc.jpg \
    -l 10 -m 10 -r 10 \
    -O --torso --data_range 0 100 --gui --asr