# 깨진 파일 이름
input_file = "preprocess2.ipynb"
# 새로 저장될 고쳐진 파일 이름
output_file = "preprocess2_1.ipynb"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 에러의 원인인 특수 공백(\xa0)을 일반 공백( )으로 모두 치환
cleaned_content = content.replace("\xa0", " ")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_content)

print("복구 완료! 이제 'preprocess2_1.ipynb'를 열어보세요.")
