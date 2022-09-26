[PYTHON] 코드로 zip 압축 풀기



1. colab에서 드라이브 연동해서 압축풀기

```python
!unzip -uq 'zip 파일 경로(확장자 포함)' -d '압축 푼 파일 저장할 경로'
```



2. url 지정해서 압축풀기

```python
import urllib
import os
import shutil
from zipfile import ZipFile


urllib.request .urlretrieve(DATA_PATH), "pill_img.zip")

with ZipFile("pill_img.zip", 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

os.rename("pill_img.zip", "img")

```

```python
## configure root folder on your gdrive
data_dir = "./data"

## custom transformer to flatten the image tensors
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        result = torch.reshape(img, self.new_size)
        return result

## transformations used to standardize and normalize the datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ReshapeTransform((-1,)) # flattens the data
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ReshapeTransform((-1,)) # flattens the data
    ]),
}

## load the correspoding folders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

## load the entire dataset; we are not using minibatches here
train_dataset = torch.utils.data.DataLoader(None,
                                            batch_size=len(image_datasets['train']),
                                            shuffle=True)
test_dataset = torch.utils.data.DataLoader(None,
                                           batch_size=len(image_datasets['val']),
                                           shuffle=True)
```



3. 

```python
import zipfile
with zipfile.ZipFile(image_dir, 'r') as existing_zip:
    existing_zip.extractall('/content/image')
```





4. 무적 코드

```python
def f_unzip(source_file, dest_path='./', enc_type='cp437', dec_type='cp949'):
    '''
    설명: 압축 파일을 지정된 디렉토리에 압축 풀기, 상태바 표시
    입력: unzip(압축 파일 경로명, 압축 풀 디렉토리 명)
    출력: 설정한 디렉토리에 압축 푼 파일 생성
    예시: src_file = '/content/drive/MyDrive/test.zip'
         des_path = '/content/img/'
         f_unzip(src_file, des_path)
    '''
    import zipfile
    import progressbar
    import time
    
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        bar = progressbar.ProgressBar(maxval=len(zipInfo)).start()
        
        for i, member in enumerate(zipInfo, start=0):
            try:
                # print(member.filename.encode(enc_type).decode(dec_type))
                member.filename = member.filename.encode(enc_type).decode(dec_type)
                zf.extract(member, dest_path)
                bar.update(i)
            except:
                print(member.filename)
                raise Exception('what?!')
    bar.finish()


src_file = '/content/drive/MyDrive/그게뭐약/alyac_label/image_598_739.zip'
des_path = '/content/img/'
f_unzip(src_file, des_path)
```





---

zip 으로 압축하기

```python
import zipfile
new_zips= zipfile.ZipFile(image_dir, 'w')
 
for folder, subfolders, files in os.walk('data/temp/'):
 
    for file in files:
        new_zips.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), 'data/temp/'), compress_type = zipfile.ZIP_DEFLATED)
 
new_zips.close()
```

