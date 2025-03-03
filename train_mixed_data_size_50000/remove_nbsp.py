import re

# 输入和输出文件路径
input_file = "Mixed_data_en-zh.zh"  # 输入文件
output_file = "Mixed_data_en-zh2.zh"  # 输出文件

def remove_nbsp(input_file, output_file):
    # 读取输入文件内容
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # 使用正则表达式去除所有的 [NBSP]
    content = re.sub(r'　', '', content)

    # 将修改后的内容写入输出文件
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"文件中的 [NBSP] 已成功去除，输出文件为 {output_file}")

# 调用函数
remove_nbsp(input_file, output_file)
