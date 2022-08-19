apt-get update
apt-get install unzip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip 
mv annotations/instances_train2017.json /tmp
mv annotations/instances_val2017.json /tmp

pip install torch --upgrade
pip install torchvision --upgrade
pip install pycocotools

mkdir ~/.aws/
cp credentials ~/.aws/