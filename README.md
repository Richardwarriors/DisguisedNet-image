DisguisedNet-image

--Dataset:
1. Breast -> Binary classification -> M & B
2. MData  -> Six classificaiton -> BG & D & N & P & S & V

--How to run:
python main.py --dataset OG --path '../Data/MData/' --output '../ModelData/MData'

Note:
Data folder contain the original data and processed RMT or AES data
ModelData folder contain the train/test/validation data processed by splitfolder

For RMT-AES:
1 set up environment: pip install numpy==1.26.1 Pillow==10.0.1 PyQt5==5.15.10 PyQt5-Qt5==5.15.11 PyQt5-sip==12.13.0 scipy==1.11.3 torch==2.1.0 torchvision==0.16.0 pycryptodome==3.19.0
2 In cryp.py:
  I cancel the random seed to make sure each time the random shuffle is different
  ```python
  if self.shuffle:
      random.shuffle(block_list)
  ```
3 In demo.py:
  with shuffle, you had to set Shuffle = True, vice versa
  ```python
  if self.rb_rmt.isChecked():
      self.encoder = RMT(image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col,image_temp.shape[2]),
                         block_size=int(self.block_size_dropdown.currentText()),
                         Shuffle=True)
  ```
