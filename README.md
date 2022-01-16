# foraminifera_detect
detect foraminiera by vgg16 using python.
also include a model i build


you need to create two folders to put the training data and testing data.

then under each folders you need too create classes folders, and put your pictures in to the folders.

before training the model, you need to check the fit setting(main.py line 58-60) is suitable for your hardware otherwise it may lead to crash.

ok that all go training your own model~



(ps the edge_detect_def and webcam_save_pic_button are for getting data)



after training and save the model as .h5 file, you can use predict.py to see the final result.

there is a function called predict(path) , you can use it to predict.

pretrained model https://drive.google.com/file/d/1Sd3jb7R30IvgXZUZZ7D89UkPBytPtmYu/view?usp=sharing



the /code file is the final product you can use it to detect Baculogypsina and Calcarina.






















coding by 蕭鴨鴨 闕承勳 陳鴻笙 杜伊凡 黃燈燈
