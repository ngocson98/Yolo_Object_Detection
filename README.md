# Yolo_Object_Detection
Tutorials Yolo_Object_Detection for custom data:
# 1. Label object
Use LabelImg -> export file txt
Note: id classes as number object.
# 2. zip file img, upload drive
# Open Google Colab
# 3. unzip file img in drive
# 4. git clone github
!git clone 'https://github.com/AlexeyAB/darknet.git' '_____PATH_____/darknet'
## Example:
!git clone 'https://github.com/AlexeyAB/darknet.git' '/content/drive/MyDrive/yolov3_v1/darknet'
# 5. Make file
!make
# 6. Upload 2 file py in file image unzip.
creating-files-data-and-name.py
##creating-train-and-test-txt-files.py
# 7. Creat folder custom_weight
Then Upload file darknet53.conv.74 inside custom_weight

# 8. Run 2 file python in colab
!python3 /content/drive/MyDrive/yolov3_v1/images/creating-files-data-and-name.py
##!python3 /content/drive/MyDrive/yolov3_v1/images/creating-train-and-test-txt-files.py

# 9. Dowload file: Makefile in darknet. Edit:
GPU=1
##CUDNN=1
OPENCV=1
##Then save, upload darknet
# 10. Dowload file: yolov3.cfg in cfg. Edit:
Line 6, 7: delete #
##Line 20: Max_batch = number class * 2000
Line 22: Step = 90%, 110% max_batch
##Find [yolo]: edit: classes, filters
Save name: yolov3_custom.cfg then upload cfg.
# 11. Train
!darknet/darknet detector train images/labelled_data.data darknet/cfg/yolov3_custom.cfg custom_weight/darknet53.conv.74 -dont_show
