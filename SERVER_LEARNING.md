# Open terminal -> login by these first steps
    ssh -p 22222 student5@ictlab.usth.edu.vn
    ssh ict14
    cd /storage/student5/depth_server
    micromamba activate da2


# Last time
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

PS C:\Users\Admin> ping 192.168.22.14

Pinging 192.168.22.14 with 32 bytes of data:
Request timed out.
Request timed out.
Request timed out.
Request timed out.

Ping statistics for 192.168.22.14:
    Packets: Sent = 4, Received = 0, Lost = 4 (100% loss),
PS C:\Users\Admin> Test-NetConnection 192.168.22.14 -Port 8000
WARNING: TCP connect to (192.168.22.14 : 8000) failed
WARNING: Ping to 192.168.22.14 failed with status: TimedOut


ComputerName           : 192.168.22.14
RemoteAddress          : 192.168.22.14
RemotePort             : 8000
InterfaceAlias         : Wi-Fi
SourceAddress          : 192.168.1.66
PingSucceeded          : False
PingReplyDetails (RTT) : 0 ms
TcpTestSucceeded       : False


# Note
- Use Storage, NO workspace, DONT HAVE SUDO!
- ict14 ip address: 192.168.22.14

- HAS DONE: env path: /storage/student5/micromamba/envs/da2

- PYTHON version server (recommend no change python3 system: On Debian 11, python3 is intentionally 3.9; many OS tools assume it)
    Python 3.9.2

- ALT WAY
    Root prefix (where micromamba keeps envs & cache):
        From your log it’s MAMBA_ROOT_PREFIX=/storage/student5/depth_server/y → env created at /storage/student5/depth_server/y/envs/da2
        /home/student5/.local/bin/micromamba
    Shell init: micromamba adds a block to ~/.bashrc so micromamba activate
    Activate env: student5@ict14:/storage/student5/depth_server$ micromamba activate da2
        (inside env): python -m pip install <packages>
    Run your server: uvicorn app:app --host 0.0.0.0 --port 8000
    micromamba shell init: export MAMBA_EXE='/storage/student5/depth_server/y/micromamba';
                           export MAMBA_ROOT_PREFIX='/storage/student5/depth_server/y';
    Open(create new and open)/edit/save/exit
        nano /storage/student5/depth_server/server.py
            Save: Ctrl+O, then Enter
            Exit: Ctrl+X

- RUN SERVER 
uvicorn server:app --host 0.0.0.0 --port 8000
    
    

# API key (my own)
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICTlOSYssq9M4OxqGIPl1w4BLkA+jwnyLocY3fsGdZdn hainh.22ba13121@usth.edu.vn

# Useful command can use for long time using
    Access in code for server:
        student5@ict14:/storage/student5$ cd depth_server
        
    Access in my storage: cd /storage/student5
    Delete file/dir: remove file: student5@ict14:/storage/student5/scratch/test$ rm note.txt
                     remove directory: student5@ict14:/storage/student5$ rm -rI /storage/student5/scratch/test

    student5@ict14:/storage/student5$ cd models                             -> go to dir models
    student5@ict14:/storage/student5/models$ ls /storage/student5/models    -> watch files in directory

    **Upload model(on local)**
    PS C:\Users\Admin> scp -P 22222 "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits.pth" student5@ictlab.usth.edu.vn:/storage/student5/models/
    depth_anything_v2_vits.pth                                                            100%   95MB   4.7MB/s   00:20

    Copy public key to server
    PS C:\Users\Admin>scp -P 22222 $env:USERPROFILE\.ssh\id_ed25519.pub student5@ictlab.usth.edu.vn:~/.ssh/tmp_key.pub

    scp -P 22222 "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_fp16.with_runtime_opt.ort" student5@ictlab.usth.edu.vn:/storage/student5/models/

# Commands understanding
	ls -ld /storage/student5: list information about the directory itself in long format
        -> Đã tạo được folder để up dataset nhưng chưa có dataset
        thay đường dẫn local cho đúng
        scp -P 22222 -r "C:\path\to\dataset"

    PROJECT=GroupProject
	mkdir -p /storage/student5/{projects,models,runs,datasets}/$PROJECT
        mkdir makes directories
        -p tells it to create any missing parents and not error if the path already exists
        The {…} part is brace expansion—Bash expands it into four paths (projects, models, runs, datasets) before running the command, and $PROJECT is a variable expansion appended at the end. Net effect: it tries to create
        /storage/student5/projects/GroupProject, /models/GroupProject, /runs/GroupProject, /datasets/GroupProject

    echo test > /storage/student5/.write_test: write "test" into write_test file 
        -> debug error 

    python3 -m venv ~/venvs/GroupProject: create environment python3

    cd /storage/student5: access to storage student5
        Debug:  pwd                                                     # should print /storage/student5
                mkdir -p scratch/test && cd scratch/test                # make directory and access to scratch/test
                echo "hello from $(hostname) at $(date)" > note.txt     # type "hello from $(hostname) at $(date)" to file note.txt
                ls -l                                                   # verify file and permissions(owner/group, size, time, permission bits)  
                cat note.txt                                            # view the contents

# Mobile app with server 
- Context
mobile → server inference → mobile
mobile app = android app
server = student5@ict14

- Architecture
    Mobile uploads an image to your server as multipart/form-data over HTTPS.
    Server runs your Depth Anything V2 model (use your .ort file with ONNX Runtime), produces a depth map.
    Server returns either:
        a 16-bit grayscale PNG (best fidelity for downstream use), and/or
        an 8-bit preview PNG (colorized) for quick display.
        PNG supports up to 16 bits per channel, so you won’t lose precision if you choose the 16-bit path.