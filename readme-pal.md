# Readme: development log


## Experiment
> work as a baseline for Zeromimic 
- [ ] Pouring tea
- [ ] Put the green package in the drawer
- [ ] Close the drawer
- [ ] Open the drawer
- [ ] Put down the green package


## TODO

### 2024-12-17
- [ ] What is different from ReKeP’s own codebase
    - [ ] Plus
        - [ ] better clustering for keypoint extraction
    - [ ] Minus
        - [ ] multiple stage: bug
        - [ ] ik cost - doensn’t matter
        - [ ] recovery
- [ ] stage decomposition  
    - store: in robot_state.json
    - load: solely in r2d2_rekep.py
    - question: keypoint following?
- [ ] Code-as-Monitor, another work with ReKeP as baseline in Omnigibson, pour tea result is 20, wherweas ReKeP official result is ~70%

### 2024-11-23
- [ ] add DINO-X into system
    - [x] prompt free detection
    - [ ] prompt guided detection
- [ ] add more visualization method
- [x] test DepthPro 
    - [x] least square method
    - [x] gaussian process
    - [x] ask GPT for help
    - [x] ask SYZ for help
- [ ] Kinemtic
    - [x] MEAM5200 note
    - [x] ECE470 note
- 
```bash
configs/ # rekep only
data/    # temp data
main.py  # original rekep for omnigibson
main_vision.py  # rekep with vision
point-draw/ # hand drawing tool
r2d2.py  # modify for r2d2
r2d2_rekep.py # deploy rekep on r2d2
r2d2.sh   # run r2d2
r2d2_vision.py # deploy rekep with vision on r2d2
vision.sh # run r2d2_vision.py
vlm_query/ # folder store vlm query
```
- [ ] Test the new tasks
- [ ] Add the new scenes
- [ ] Add the new objects
- [ ] Add new camera views
- [ ] 3D visualization of RGBD
- [ ] extristic and intrinsic camera calibration
- [ ] 3D bounding box visualization? 
- [ ] add grounding dino into system
