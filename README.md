# keras-superresolution
Implementation of four different deep learning models for super-resolution.

The implemented models are:

- RGB Fast Super Resolution Convolutional Neural Netowork (FSRCNN, based on: https://link.springer.com/chapter/10.1007/978-3-319-46475-6_25);
- Encoder-Decoder architecture (not really based on anything, ED architectures are just good at this stuff);
- Deep residual encoder-decoder with skip connections (shallow REDNET, based on: https://arxiv.org/abs/1606.08921)
- Multi Scale Laplacian Super Resolution Network (MS-LapSRN, based on: https://ieeexplore.ieee.org/abstract/document/8434354/)

The models are trained on traffic signs (the Traffic Sign Recogntion Database - http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)

You can either use the pretrained models for super-resolution (although they only work on the TSRD), or train them again from scratch on a different dataset (like DIV2k). 

Helper functions include some stuff for inference time measurement, metrics (PSNR and SSIM), and generating LR images. A cubify helper function is also included, which divides larger pictures into smaller ones (for training on different datasets). 


