# 2D acoustic holography using iterative learning method with physics model
The code for the paper: Unsupervised Learning with Physics Based Mutual Enhancement for Real-time Acoustic Holography


** The paper has been submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS) **
## Contact
- The first author: zhongchx@shanghaitech.edu.cn
- Corresponding author: liusong@shanghaitech.edu.cn


## Environment Setup
Make sure your computer has the following software and tools installed:
python 3.7.11
the required package can be installed by commands ``` pip install -r requirements.txt ```
It is recommended to run the code in a virtual environment to isolate dependencies between different projects.


## Code
- Runnable code is in the .py file ``` Acousholo_main.py ```
- Dataset path is ``` ./dataset/ ```
- Important packages are listed in ``` requirements.txt ```
- The Angular Spectrum Method (ASM), acoustic field propagation function and other auxiliary function is in the .py file ``` auxiliaryfunc.py ```
- The iterative learning framework is in the .py file ``` Acousholo_trainingframework.py ```
- The evaluation function for validation and testing is in the .py file ``` Acousholo_evaluation.py ```
- The loss function used for network training is in the .py file ``` Acousholo_lossfunc.py ```

- A saved running result is in ``` Results_newOrgamized ``` folder with subfolder named by ``` Time parameter ```
    + curves: save training and validation curves in a certain training epoch
    + model_path: save Best and Final .pth file
    + testmetrics: save metric values on testing set
    + trainparams: save the training parameters
    + trrData: plot reconstructed amplitude hologram, indicating its max reconstructed amplitude value in the file name
               save target holo, retrieved phs holo, reconstructed holo pixel-wise value, indicating its reconstruction PSNR value in the file name
    + visualize: visualize test results (target amplitude hologram, reconstructed amplitude hologram, predicted POH and reconstructed phase hologram)


## Train
You can customize your own training case by setting the following parameters in ``` Acousholo_main.py ``` file which have been highlighted by
``` ============================ You can change parameters here ============================ ``` and
```========================================================================================```.
## Test
The test code is automatically completed after the training, which is shown in the at the end of the .py file ``` Acousholo_main.py ```