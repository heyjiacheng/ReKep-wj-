python main_vision.py \
    --instruction "Move the yellow pineapple on the roll to the frying pan." \
    --obj_list "yellow pineapple,frying pan" \
    --data_path "data/rgbd/pan-complex/" \
    --frame_number 33 \
    --visualize
python main_vision.py \
    --instruction "Fold the cloth step by step." \
    --data_path "data/rgbd/cloth-hack-1/" \
    --frame_number 10 \
    --visualize