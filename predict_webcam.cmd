call conda activate ldc
call yolo predict model=torch_files/yolov8n-smartphone.pt source=0 show=True line_thickness=2 conf=0.6
pause