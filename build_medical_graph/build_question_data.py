import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split

def entity_extract(pred):
    if not pred:
        return []

    cur_entity = None
    res = []
    start_idx,end_idx = 0,0

    for i,pre_single in enumerate(pred):
        pred_start_B = pre_single.startswith('B') # 实体标签开头B
        pred_entity = pre_single.split('_')[-1]

        if cur_entity:
            if pred_start_B or cur_entity != pred_entity:
                res.append({
                    'st_idx':start_idx,
                    'end_idx':i,
                    'label':cur_entity
                })
                cur_entity = None
        if pred_start_B:
            start_idx = i
            cur_entity = pred_entity
    if cur_entity:
        res.append({
            'st_idx':start_idx,
            'end_idx':len(pred),
            'label':cur_entity,
        })
    return res

def build_entity_csv():
    # 把文档中涉及的所有实体提取出来
    with open('./data/temp/data/dev.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts = ''
    labels = []
    content = []
    label_class = []
    for line in lines:
        if line != '\n':
            try:
                text,label = line.strip().split(' ')
            except ValueError:
                continue
            texts += text
            labels.append(label)
        else:
            res = entity_extract(labels)
            st_idx = res[0]['st_idx']
            end_idx = res[0]['end_idx']
            entity = texts[st_idx:end_idx]
            texts = ''
            labels = []
            if entity in content:
                continue
            content.append(entity)
            label_class.append(res[0]['label'])

    csv_path = './data/temp/entity_extract.csv'
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['entity','label'])
        df['entity'] = content
        df['label'] = label_class
        df.to_csv(csv_path,index=False)
    else:
        df = pd.read_csv(csv_path)
        new_df = pd.DataFrame({'entity': content, 'label': label_class})
        df = pd.concat([df, new_df], ignore_index=True)
        df.drop_duplicates(subset='entity', keep='first', inplace=True)
        df.to_csv(csv_path, index=False)

def build_question_file():
    with open('data/temp/治愈率.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    entity_df = pd.read_csv('./data/temp/entity_extract.csv')

    label_class = entity_df['label']
    diseases = entity_df[entity_df['label'] == 'disease']['entity'].tolist()

    questions = []
    labels = []
    for name in diseases:
        for line in lines:
            text,label = line.strip().split('-')
            questions.append(text.replace('[disease]',name))
            labels.append(label)

    df = pd.DataFrame({'text':questions,'label':labels})
    df.to_csv('./data/temp/questions/治愈率.csv',index=False)


if __name__ == '__main__':

    # build_question_file()

    dir_list = os.listdir('./data/temp/questions')

    new_df = pd.DataFrame(columns=['text','label'])
    for item in dir_list:
        print(item)
        path = './data/temp/questions/' + item
        df = pd.read_csv(path)
        print(len(df))
        new_df = pd.concat([df,new_df], ignore_index=True)

    new_df.to_csv('./data/all_questions.csv',index=False)

    train_data,test_data = train_test_split(new_df,train_size=0.7)
    train_df = pd.DataFrame(train_data,columns=['text','label'])
    train_df.to_csv('./data/self_train.csv',index=False)

    test_df = pd.DataFrame(test_data, columns=['text', 'label'])
    test_df.to_csv('./data/self_test.csv',index=False)

    data = pd.read_csv('./data/all_questions.csv')
    print(data['label'].value_counts())

    pass

