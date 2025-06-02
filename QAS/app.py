from flask import Flask, redirect, url_for, request, render_template, jsonify
import pickle
import pandas as pd
import json, datetime
from transformers import PhobertTokenizer, BertForQuestionAnswering
import torch
from collections import Counter

data = pd.read_excel('data/train_data.xlsx')
file_path = "data/chat_history.json"

# Đọc dữ liệu từ file JSON
try:
   with open(file_path, "r", encoding='utf-8') as file:
      chat_history = json.load(file)
except FileNotFoundError:
   chat_history = [] 

# Load mô hình
url = 'c:/Users/ngohu/Documents/DA2/QAS/data/ModelAI/'
def loadmodel(model):
   with open(model, 'rb') as file:
      model = pickle.load(file)
   return model

classifier, vectorizer = loadmodel(url + 'classifier.pkl'), loadmodel(url + 'vectorizer.pkl')
model_name = "ngohuule16012000/QASBert-THPT"
tokenizerBert = PhobertTokenizer.from_pretrained(model_name)
modelBert = BertForQuestionAnswering.from_pretrained(model_name)

# tính toán f1 score
def calculate_f1(true_answer, pred_answer):
   if type(true_answer) == int:
      true_answer_tokens = true_answer
   else:
      true_answer_tokens = true_answer.lower().split()
   if type(pred_answer) == int:
      pred_answer_tokens = pred_answer
   else:
      pred_answer_tokens = pred_answer.lower().split()
   common_tokens = Counter(str(true_answer_tokens)) & Counter(str(pred_answer_tokens))

   # Số lượng tokens chung giữa câu trả lời dự đoán và đúng
   num_same = sum(common_tokens.values())

   if num_same == 0:
      return 0

   precision = 1.0 * num_same / len(str(pred_answer_tokens))
   recall = 1.0 * num_same / len(str(true_answer_tokens))
   f1 = (2 * precision * recall) / (precision + recall)

   return f1

# dự đoán câu trả lời
def predict_answer(question, contents, model, tokenizer):
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # model.to(device)
   model.eval()

   f1_scores = []
   answers = []

   for content in contents:
      # Tokenize input
      inputs = tokenizer.encode_plus(question, content,  return_tensors="pt", 
         padding=True, 
         truncation= 'only_second',
         max_length=258
      )
      
      # Make prediction
      outputs = model(**inputs)
      start_scores, end_scores = outputs.start_logits, outputs.end_logits

      # Get the most likely answer
      start_index = torch.argmax(start_scores)
      end_index = torch.argmax(end_scores) + 1 # Add 1 because the end index is inclusive'
      # start_scores_list = start_scores.tolist()
      # end_scores_list = end_scores.tolist()
      # start_index = start_scores_list[0].index(max(start_scores_list[0]))
      # end_index = end_scores_list[0].index(max(end_scores_list[0])) + 1 # Add 1 because the end index is inclusive
      # Decode the answer from the tokens
      predicted_answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index], skip_special_tokens=True)
      answers.append(predicted_answer)

      if predicted_answer != "":
         true_answers = data[data['context'] == content]['answers'].tolist() 
         for true_answer in true_answers:
            f1 = calculate_f1(true_answer, predicted_answer)
            # if f1 != 0:
            #    f1_scores.append(f1)
            f1_scores.append(f1)
   
   if len(f1_scores) != 0:
      average_f1_score = sum(f1_scores) / len(f1_scores)
   else:
      average_f1_score = 0.0
   print("f1", f1_scores, average_f1_score)
   return answers, average_f1_score

app = Flask(__name__)

def chat():
   colors = ['#EE0000','#DD0000','#CC0000','#BB0000','#AA0000','#990000','#880000','#770000','#660000','#550000','#440000','#330000','#220000','#110000','#CCCC66','#CCCC33','#CCCC00','#99CCFF','#99CCCC','#99CC99','#99CC66','#99CC33','#99CC00','#66CCFF','#66CCCC','#66CC99','#66CC66','#66CC33','#66CC00','#33CCFF','#33CCCC','#33CC99','#33CC66','#33CC33','#33CC00','#00CCFF','#00CCCC','#33CC66','#33CC33','#00CC99','#00CC66','#00CC33','#00CC00','#FF99FF','#FF99CC','#FF9999','#FF9966','#FF9933','#FF9900','#CC99FF','#CC99CC','#CC9999','#CC9966','#CC9933','#CC9900','#9999FF','#9999CC','#999999','#999966','#999933','#999900','#6699FF','#6699CC','#669999','#669966','#669933','#669900','#3399FF','#3399CC','#339999','#339966','#339933','#339900','#0099FF','#0099CC','#009999','#009966','#009933','#009900','#FF66FF','#FF66CC','#FF6699','#FF6666','#FF6633','#FF6600','#CC66FF','#CC66CC','#CC6699','#CC6666','#CC6633','#CC6600','#9966FF','#9966CC','#996699','#996666','#996633','#996600','#6666FF','#6666CC','#666699','#666666','#666633','#666600','#3366FF','#3366CC','#336699','#336666','#336633','#336600','#0066FF','#0066CC','#006699','#006666','#006633','#006600','#FF33FF','#FF33CC','#FF3399','#FF3366','#FF3333','#FF3300','#CC33FF','#CC33CC','#CC3399','#CC3366','#CC3333','#CC3300','#9933FF','#9933CC','#993399','#993366','#993333','#993300','#6633FF','#6633CC','#663399','#663366','#663333','#663300','#3333FF','#3333CC','#333399','#333366','#333333','#333300','#0033FF','#FF3333','#0033CC','#003399','#003366','#003333','#003300','#FF00FF','#FF00CC','#FF0099','#FF0066','#FF0033','#FF0000','#CC00FF','#CC00CC','#CC0099','#CC0066','#CC0033','#CC0000','#9900FF','#9900CC','#990099','#990066','#990033','#990000','#6600FF','#6600CC','#660099','#660066','#660033','#660000','#3300FF','#3300CC','#330099','#330066','#330033','#330000','#0000FF','#0000CC','#000099','#000066','#000033','#00FF00','#00EE00','#00DD00','#00CC00','#00BB00','#00AA00','#009900','#008800','#007700','#006600','#005500','#004400','#003300','#002200','#001100','#0000FF','#0000EE','#0000DD','#0000CC','#0000BB','#0000AA','#000099','#000088','#000077','#000055','#000044','#000022','#000011','#33CCCC','#33CC99','#33CC66','#33CC33','#33CC00','#00CCFF','#00CCCC','#33CC66','#33CC33','#00CC99','#00CC66','#00CC33','#00CC00','#FF99FF','#FF99CC','#FF9999','#FF9966','#FF9933','#FF9900','#CC99FF','#CC99CC','#CC9999','#CC9966','#CC9933','#CC9900','#9999FF','#9999CC','#999999','#999966','#999933','#999900','#6699FF','#6699CC','#669999','#669966','#669933','#669900','#3399FF','#3399CC','#339999','#339966','#339933','#339900','#0099FF','#0099CC','#009999','#009966','#009933','#009900','#FF66FF','#FF66CC','#FF6699','#FF6666','#FF6633','#FF6600','#CC66FF','#CC66CC','#CC6699','#CC6666','#CC6633','#CC6600','#9966FF','#9966CC','#996699','#996666','#996633','#996600','#6666FF','#6666CC','#666699','#666666','#666633','#666600','#3366FF','#3366CC','#336699','#336666','#336633','#336600','#0066FF','#0066CC','#006699','#006666','#006633','#006600','#FF33FF','#FF33CC','#FF3399','#FF3366','#FF3333','#FF3300','#CC33FF','#CC33CC','#CC3399','#CC3366','#CC3333','#CC3300','#9933FF','#9933CC','#993399','#993366','#993333','#993300','#6633FF','#6633CC','#663399','#663366','#663333','#663300','#3333FF','#3333CC','#333399','#333366','#333333','#333300','#0033FF','#FF3333','#0033CC','#003399','#003366','#003333','#003300','#FF00FF','#FF00CC','#FF0099','#FF0066','#FF0033','#FF0000','#CC00FF','#CC00CC','#CC0099','#CC0066','#CC0033','#CC0000','#9900FF','#9900CC','#990099','#990066','#990033','#990000','#6600FF','#6600CC','#660099','#660066','#660033','#660000','#3300FF','#3300CC','#330099','#330066','#330033','#330000','#0000FF','#0000CC','#000099','#000066','#000033','#00FF00','#00EE00','#00DD00','#00CC00','#00BB00','#00AA00','#009900','#008800','#007700','#006600','#005500','#004400','#003300','#002200','#001100','#0000FF','#0000EE','#0000DD','#0000CC','#0000BB','#0000AA','#000099','#000088','#000077','#000055','#000044','#000022'];
   texts = data['Title'].drop_duplicates().tolist()
   numbers = data.groupby('Title').count()['context'].tolist()
   sorted_pairs = sorted(zip(numbers, texts), key=lambda x: x[0], reverse=True)
   nums, contents = zip(*sorted_pairs)
   print(len(contents))
   flag = 1
   n = nums[0]
   cols = []
   cols.append(colors[0])
   for i in range(1, len(nums)):
      if nums[i] == n:
         cols.append(colors[flag])
      else:
         n = nums[i]
         flag += 1
         cols.append(colors[flag])

   buttons = zip(nums, contents, cols)
   return render_template('chat.html', buttons=buttons)
app.add_url_rule('/', 'chat', chat, methods = ['POST', 'GET'])

# lịch sử chatbox
def hischatbox():
   # Đọc file JSON
   with open('C:/Users/ngohu/Documents/DA2/QAS/data/chat_history.json', encoding='utf-8') as json_file:
      messages = json.load(json_file)

   sorted_messages = sorted(messages, key=lambda x: datetime.datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S.%f"), reverse=True)
    
   # Render template và truyền dữ liệu JSON vào template
   return render_template('admin.html', messages=sorted_messages)
app.add_url_rule('/admin', 'admin', hischatbox)

# Phân loại câu hỏi
def predict_intent(question):
   question_vect = vectorizer.transform([question])  # Chuyển đổi câu hỏi thành vector đặc trưng
   intent = classifier.predict(question_vect)
   return intent

# Tạo nội dung cho json
def chatJson(user, bot):
   return {"user_chat": user, "bot_chat": bot, "timestamp": str(datetime.datetime.now())}

# Thêm tính chuyên nghiệp cho câu trả lời
def fixString(chatstring):
   # Loại bỏ khoảng trắng, dấu phẩy, dấu chấm ... ở đầu chuỗi
   chatstring = chatstring.lstrip(' .,:?')
   # Viết hoa chữ cái đầu tiên của chuỗi
   chatstring = chatstring.capitalize()
   # Loại bỏ dấu phẩy, chấm phẩy ... ở cuối chuỗi và thêm dấu chấm
   chatstring = chatstring.rstrip(' ,;?:.') + '.'
   return chatstring

def prediction_message():
   message = request.form["message"]

   prediction = predict_intent(message)[0]
   contents = data[data['Title'] == prediction]['context'].drop_duplicates().tolist()
   
   answers, f1_scores = predict_answer(message, contents, modelBert, tokenizerBert)

   predict = " ".join([fixString(i) for i in set(filter(None, answers))])
   chat_history.append(chatJson(message, predict))

   with open(file_path, "w", encoding='utf-8') as file:
      json.dump(chat_history, file, ensure_ascii = False, indent=4)
   return jsonify({"prediction": predict, "f1scores": f1_scores})
app.add_url_rule('/prediction_message', 'prediction_message', prediction_message, methods = ['POST'])

if __name__ == '__main__':
   app.run(debug = True)


