# **DS102.L11.CNCL** _Object Detection_

## **_A. Huong dan cai dat_**

### **I. Setup environment**

#### 1. Cai dat Linux Terminal tren Windows

- Link: [Microsoft](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- Tai Ubuntu 20.04 tren Windows Store

#### 2. Update terminal

- > sudo apt update
- > sudo apt upgrade
- > sudo apt install gcc

#### 3. Cai dat anaconda tren ubuntu terminal

- Tai anaconda cho ubuntu (file dang .sh)
- > chmod +x ten_file_anaconda.sh
- > ./ten_file_anaconda.sh
- use conda init -> yes
- Tat terminal hien tai roi mo lai
- > conda update --all

### 4. Tao environment de lam viec

- > conda create --name ds102 tensorflow-gpu
- > conda activate ds102

### 5. Cai package de lam viec

- > conda install matplotlib
- > conda install pandas
- > pip install pickle5
- > pip install torch torchvision
- > conda install scikit-learn
- > pip install opencv-python

### **II. Cai dat Latex**

- Mo terminal
- > sudo apt install texlive
- > sudo apt install latexmk

### **III. Set up VSCode**

#### 1. Download + install vscode

- Link: [VSCode](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

#### 2. Cai dat extension

- LaTex Workshop
- Live Share
- Remote - WSL

## **B. Noi dung can tim hieu**

### **I. Image Processing**

#### 1. Normalizing (Range 0-1)

#### 2. Random Cropping

#### 3. Random Flipping

#### 4. Cut out regularization

#### 5. Image Augmentation (Optional)

### **II. Hyperparameter Tuning**

#### 1. Grid Search

#### 2. Random Search

#### 3. Evolutionary Search

#### 4. Bisection

### **III. Evaluation Metrics**

#### 1. K-Fold Stratified Cross-validation

#### 2. IoU

#### 3. AP

### **IV. Machine-learning Models for Object Detection**

#### 1. Random Forest

#### 2. SVM

### **V. SOTA Deep-learning Models**

#### 1. YOLO Algorithm

#### 2. R-CNN

#### 3. Sliding Window

## **_C. Road Map_**

### **Stage I**

- Tien xu ly anh
- Code xong phan log du lieu moi experimental run
- Code xong phan luu du model moi experimental run
- Code chay ko bug
- Viet report 25%

### **Stage II**

- Chay xong cac thi nghiem
- Plot cac figure danh gia
- Build cac table de danh gia
- Viet report 75%

### **Stage III**

- Lam slide seminar
- Viet report 100%
- Code demo ket qua
