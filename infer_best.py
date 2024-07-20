import json
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report,precision_score, recall_score, f1_score


import dataProcessor

device="cuda:0"
plm_path = 'plmpath'

label2id={'个人事例': 0, '中心论点': 1, '事例史实': 2, '公理规律': 3, '其他': 4, '分论点': 5, '名人名言': 6, '谚语俗语': 7, '重申论点': 8, '阐述': 9} 
id2label = {0: '个人事例', 1: '中心论点', 2: '事例史实', 3: '公理规律', 4: '其他', 5: '分论点', 6: '名人名言', 7: '谚语俗语', 8: '重申论点', 9: '阐述'}



def compute_2_f1(true_labels, predicted_labels):
    

    fine2couse={'中心论点':'论点',
                '分论点':'论点',
                '重申论点':'论点',

                '事例史实':'论据',
                '个人事例':'论据',
                '名人名言':'论据',
                '谚语俗语':'论据',
                '公理规律':'论据',
                
                '阐述':'论证',
                
                '其他':'其他'
    }
    fine_f1 = f1_score(y_true =true_labels, y_pred =predicted_labels, average='micro')

    c_pre,c_true=[],[]
    for true,pre in zip(true_labels, predicted_labels):
        c_true.append(fine2couse[true])
        c_pre.append(fine2couse[pre])
    corse_f1 =f1_score(y_true =c_true, y_pred =c_pre, average='micro')

    return fine_f1,corse_f1*0.5+fine_f1*0.5




def just_infer(model_paths,json_dir='combine_NewFolder.json'):
    val_data = dataProcessor.DataSet(json_dir='/home/wangchuhan/file/NLPCC2024/Data/test_data.json',
          max_seqlen=37,plm_path='/home/wangchuhan/file/NLPCC2024/ChinesePLM/chinese_roberta_wwm_large_ext_pytorch')

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)
    print(id2label)

    with open('/home/wangchuhan/file/NLPCC2024/Data/test_data.json', 'r') as file:
        待写入 = json.load(file) 

    for i,item in enumerate(待写入):
        new_sents = []
        for j,sen in enumerate(待写入[i]["sents"]) : 
                new_sents.append({"sentText":sen,
                              'pro_xi':[],
                              'decoded_encodings':[]
                })
        待写入[i]["sents"] =  new_sents
    '''with open('xxxxxx.json', 'w',encoding='utf-8') as file:
        json.dump(待写入, file,indent=5,ensure_ascii=False)
    exit()'''


    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/home/wangchuhan/file/NLPCC2024/ChinesePLM/chinese_roberta_wwm_large_ext_pytorch')

    for model_dir in model_paths:
        model = torch.load(model_dir)
        model = model.to(device)
        填充=0
        count = 0
        with torch.no_grad():
            data_index = 0
            for item in tqdm(val_dataloader):
                count +=1
                
                input_ids,input_att = item['input_ids'],item['attention_mask']
                                
                #print(item["essay_ID"],"Decoded sequence: ", decoded_encodings)

                input_ids,input_att = input_ids.to(device), input_att.to(device),
                drop =False

                #print(input_ids.shape,input_att.shape,paragraph_nums)
                output=model(input_ids,input_att,drop=drop)#,para_id=paragraph_nums)
                #output=torch.randint(0,2, (42, 10))
                p_label =torch.tensor([]).to(device)
                # 遍历每一行
                for i,seq_len in enumerate(item['seq_len']):
                    prow =output[i][:seq_len]
                    if seq_len > len(prow):
                        print(prow,prow.shape)
                        for _ in range(seq_len - len(prow)):
                            prow=torch.cat((prow,torch.tensor([[0,0,0,0,0,0,0,0,0,1.0]]).to(device)))#阐述
                        填充 +=seq_len-len(prow)
                    p_label=torch.cat((p_label,prow))
                #print(item['seq_len'],p_label.shape)#torch.Size([14, 10])

                for j,sentence in enumerate(待写入[data_index]["sents"]):
                    if j < len(item['input_ids'][0]):
                        decoded_encodings=tokenizer.decode(item['input_ids'][0][j])
                    else: decoded_encodings=""
                    待写入[data_index]["sents"][j]['pro_xi'].append(p_label.cpu().tolist()[j]) 
                    print(待写入[data_index]["sents"][j]['pro_xi'])
                    待写入[data_index]["sents"][j]['decoded_encodings'].append(decoded_encodings)

    
                data_index +=1
        with open(model_dir+'.json', 'w',encoding='utf-8') as file:
            json.dump(待写入, file,indent=5,ensure_ascii=False)
    count = 0
    for data_index,item in enumerate(待写入):
        count +=1
        
        for j,sentence in enumerate(待写入[data_index]["sents"]):
            real_proxi =torch.tensor(sentence['pro_xi'][0])
            for jjjpro in range(1,len(sentence['pro_xi'])):
                real_proxi+=torch.tensor(sentence['pro_xi'][jjjpro])
            labelid = real_proxi.argmax().cpu().tolist()
            print(real_proxi,labelid)
            待写入[data_index]["sents"][j]["fine_sent_type"]=id2label[labelid]
    with open(json_dir, 'w',encoding='utf-8') as file:
        json.dump(待写入, file,indent=5,ensure_ascii=False)
    




if __name__ == "__main__":
    
    paths = ['42+roberta+mlm+att(37x256)/epoch2.pth',
             '1234+roberta+mlm+att(37x256)/epoch2.pth',
             '3407+roberta+mlm+att(37x256)/epoch2.pth']

    just_infer(paths)

