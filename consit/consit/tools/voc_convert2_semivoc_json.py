import json
import xml.etree.ElementTree as ET
import os
import argparse
import shutil

CLASSES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field',
              'roundabout', 'harbor',
              'swimming-pool', 'helicopter']
def convert(args):
        print("Generating consit-style data dir")
        if not os.path.exists(args.output):
                os.mkdir(args.output)

        if not os.path.exists(os.path.join(args.output,'prepared_annos/Industry/annotations/full')):
                os.makedirs(os.path.join(args.output,'prepared_annos/Industry/annotations/full'))

        if not os.path.exists(os.path.join(args.output,'images/full')):
                os.makedirs(os.path.join(args.output,'images/full'))
                with open(os.path.join(args.input,'VOC2007/ImageSets/Main/trainval.txt'), 'r') as f:
                    lines = f.readlines()
                    with open(os.path.join(args.output, 'prepared_annos/Industry/train_list.txt'),'w') as f1:
                        for line in lines:
                            f1.write(line.strip()+'.png'+'\n')
                            shutil.copyfile(os.path.join(args.input, 'VOC2007/JPEGImages/', line.strip()+'.png'), os.path.join(args.output, 'images/full',line.strip()+'.png'))

        if not os.path.exists(os.path.join(args.output,'valid_images/full')):
                os.makedirs(os.path.join(args.output,'valid_images/full'))
                with open(os.path.join(args.input,'VOC2007/ImageSets/Main/test.txt'), 'r') as f:
                    lines = f.readlines()
                    with open(os.path.join(args.output, 'prepared_annos/Industry/valid_list.txt'),'w') as f1:
                        for line in lines:
                            f1.write(line.strip()+'.png'+'\n')
                            shutil.copyfile(os.path.join(args.input, 'VOC2007/JPEGImages/', line.strip()+'.png'), os.path.join(args.output, 'valid_images/full',line.strip()+'.png'))

        if not os.path.exists(os.path.join(args.output, 'unlabel_prepared_annos/Industry/annotations/full/')):
                os.makedirs(os.path.join(args.output, 'unlabel_prepared_annos/Industry/annotations/full/'))

        if not os.path.exists(os.path.join(args.output,'unlabel_images/full')):
                os.makedirs(os.path.join(args.output,'unlabel_images/full'))
                lines = os.listdir("/public/home/lsy/myj/code_dsl/DOTA/unlabel_train_clip")
                with open(os.path.join(args.output, 'unlabel_prepared_annos/Industry/voc12_trainval.txt'),'w') as f1:
                    for line in lines:
                        f1.write(line.strip()+'\n')
                        shutil.copyfile(os.path.join('/public/home/lsy/myj/code_dsl/DOTA/unlabel_train_clip', line.strip()), os.path.join(args.output, 'unlabel_images/full',line.strip()))


        ## for  annotations
        print("Converting original voc annotations to consit-style annotations")
        ori_catid={}
        with open(os.path.join(args.output,'mmdet_category_info.json'),'w') as f:
                cat_info = {}
                cat_info['cat2id']={}
                cat_info['id2cat']={}
                for i in range(len(CLASSES)):
                        if CLASSES[i] not in cat_info['cat2id'].keys():
                                cat_info['cat2id'][CLASSES[i]] = i
                                cat_info['id2cat'][str(i)] = CLASSES[i]
                        #ori_catid[str(data_val['categories'][i]['id'])]=data_val['categories'][i]['name']
                cat_info['cat2id']['背景']=int(len(CLASSES))
                cat_info['id2cat'][str(int(len(CLASSES)))]='背景'
                json.dump(cat_info, f, indent=4,ensure_ascii=False)
        # init val list
        with open(os.path.join(args.output,'prepared_annos/Industry/valid_list.txt'),'r') as f:
                for line in f.readlines():
                        name = line.strip()
                        with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full',name)+'.json','w') as f1:
                                res={}
                                res["imageName"]='full/' + name
                                res["targetNum"]=0
                                res["rects"]=[]
                                res["tags"]=[]
                                res["masks"]=[]
                                tree = ET.parse(os.path.join(args.input, 'VOC2007/Annotations/', name.replace('png','xml')))
                                root = tree.getroot()
                                bboxes = []
                                labels = []
                                bboxes_ignore = []
                                labels_ignore = []
                                for obj in root.findall('object'):
                                    Name = obj.find('name').text
                                    if Name not in CLASSES:
                                        continue
                                    difficult = obj.find('difficult')
                                    difficult = 0 if difficult is None else int(difficult.text)
                                    bnd_box = obj.find('bndbox')
                                    # TODO: check whether it is necessary to use int
                                    # Coordinates may be float type
                                    bbox = [
                                        # int(float(bnd_box.find('xmin').text))-1,
                                        # int(float(bnd_box.find('ymin').text))-1,
                                        # int(float(bnd_box.find('xmax').text))-1,
                                        # int(float(bnd_box.find('ymax').text))-1
                                        int(float(bnd_box.find('xmin').text)) ,
                                        int(float(bnd_box.find('ymin').text)) ,
                                        int(float(bnd_box.find('xmax').text)) ,
                                        int(float(bnd_box.find('ymax').text))
                                    ]
                                    ignore = False
                                    if difficult or ignore:
                                        continue
                                    else:
                                        res['targetNum'] +=1
                                        res['rects'].append(bbox)
                                        res['tags'].append(Name)
                                        res['masks'].append([])
                                json.dump(res, f1, indent=4, ensure_ascii=False)
        # init train list
        with open(os.path.join(args.output,'prepared_annos/Industry/train_list.txt'),'r') as f:
                for line in f.readlines():
                        name = line.strip()
                        with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full',name)+'.json','w') as f1:
                                res={}
                                res["imageName"]='full/' + name
                                res["targetNum"]=0
                                res["rects"]=[]
                                res["tags"]=[]
                                res["masks"]=[]
                                tree = ET.parse(os.path.join(args.input, 'VOC2007/Annotations/', name.replace('png','xml')))
                                root = tree.getroot()
                                bboxes = []
                                labels = []
                                bboxes_ignore = []
                                labels_ignore = []
                                for obj in root.findall('object'):
                                    Name = obj.find('name').text
                                    if Name not in CLASSES:
                                        continue
                                    difficult = obj.find('difficult')
                                    difficult = 0 if difficult is None else int(difficult.text)
                                    bnd_box = obj.find('bndbox')
                                    # TODO: check whether it is necessary to use int
                                    # Coordinates may be float type
                                    bbox = [
                                        # int(float(bnd_box.find('xmin').text))-1,
                                        # int(float(bnd_box.find('ymin').text))-1,
                                        # int(float(bnd_box.find('xmax').text))-1,
                                        # int(float(bnd_box.find('ymax').text))-1
                                        int(float(bnd_box.find('xmin').text)) ,
                                        int(float(bnd_box.find('ymin').text)) ,
                                        int(float(bnd_box.find('xmax').text)) ,
                                        int(float(bnd_box.find('ymax').text))
                                    ]
                                    ignore = False
                                    if difficult or ignore:
                                        continue
                                    else:
                                        res['targetNum'] +=1
                                        res['rects'].append(bbox)
                                        res['tags'].append(Name)
                                        res['masks'].append([])
                                json.dump(res, f1, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--input',type=str,default=None,help="VOCdevkit path: /gruntdata2/tcguo/VOCdevkit")
    parser.add_argument('--output',type=str,default=None,help="consit-style semivoc data dir path: /gruntdata1/bhchen/factory/data/semivoc")
    args= parser.parse_args()
    convert(args)
