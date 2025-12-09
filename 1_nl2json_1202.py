'''
调整一下prompt，把一些说明放到system_prompt
'''

PROMPT = '''
You are an expert assistant for a household robot. Your task is to convert natural language instructions into a structured JSON format representing Signal Temporal Logic (STL) constraints.

The position and size of each box are described by the coordinates: (x_start, x_end, y_start, y_end). The values x_start, x_end define the box's horizontal bounds, and y_start, y_end define the vertical bounds.

The size of the environmental site is (0,10,0,10). 
Scene objects:
['name': 'RestRoom', 'color': 'yellow', 'position and size': (0, 3, 7.5, 10.0)]
['name': 'MasterBedroom', 'color': 'green', 'position and size': (7, 10, 5.5, 10.0)]
['name': 'RestRoom2', 'color': 'pink', 'position and size': (7, 10, 2, 4)]
['name': 'ExerciseRoom', 'color': 'deepblue', 'position and size': (4, 6, 8, 10)]
['name': 'LivingRoom', 'color': 'blue', 'position and size': (2, 5, 3, 6)]
['name': 'Kitchen', 'color': 'cyan', 'position and size': (0, 1, 0, 2)]
['name': 'DiningRoom', 'color': 'purple', 'position and size': (2, 4, 0, 1)]
['name': 'Bedroom', 'color': 'red', 'position and size': (5, 10, 0, 2)]


You must output a single valid JSON object with the following structure:
{
  "initial_condition": {
    "position": [1, 3] // Fixed as [1, 3]
  },
  "temporal_constraints": [
    // List of constraints (sequence, choice)
  ],
  "global_constraints": [
    // List of global constraints (always_avoid, always_satisfy)
  ],
  "time_bounds": {
    "total": <number> or <interval list>// Optional total time limit
  }
}

Constraint Types & Schema
1. sequence: Visit rooms in a specific order.
  - type: "sequence"
  - tasks: List of task objects, where each task has:
    - target: Room name.
    - time_bound: Time limit or time interval for this step (number or list, optional).
    - actions: List of actions (e.g., [{"wait": 5}], optional).
2. choice: Choose one branch from multiple options.
  - type: "choice"
  - options: List of option objects, where each option has:
    - room: Target room name.
    - time_bound: Time limit or time interval for this step (number or list, optional).
    - actions: List of actions.
3. always_avoid / always_satisfy: Global invariants.
  - type: "always_avoid" or "always_satisfy"
  - condition: Object describing the condition.
    - room: Room name.

Examples
Input:
"Visit all cyan, purple, and red rooms in any order, but ensure you enter the LivingRoom before the MasterBedroom. After visiting all these rooms, proceed to either the RestRoom or the ExerciseRoom. If you choose the RestRoom, you must complete your journey within 20 seconds; if you choose the ExerciseRoom, you must wait there for 5 seconds before completing your journey within 25 seconds. Throughout your journey, avoid entering any pink-colored rooms."
Response:
{
  "initial_condition": { "position": [1, 3] },
  "temporal_constraints": [
    {
      "type": "sequence",
"tasks": [
        {
          "target": "Kitchen",
        },
      {
          "target": "DiningRoom",
        },
        {
          "target": "Bedroom",
        },
        {
          "target": "LivingRoom"
        },
  {
          "target": "MasterBedroom"
        }
      ]
    }
  ]
,
    {
      "type": "choice",
      "options": [
        { "room": "RestRoom", "time_bound": 20},
        { "room": "ExerciseRoom", "time_bound": 25, "actions": [
  {
              "type": "wait",
              "duration": 5
            }] }
      ]
    }
  ],
  "global_constraints": [
    { "type": "always_avoid", "condition": { "room": "RestRoom2"} }
  ],
  "time_bounds": { "total": 25 }
}

'''

import json
import random
import re
from openai import *
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
import time
'''
调用llm
'''
client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-MZT6pRkO220urRvuW4IdSkA3zfXkOAKelUUXqSJgmRUJSIZU",
    base_url = "https://api.chatanywhere.tech/v1"
)

def call_llm(nl):
    prompt_nl = f'''
Input:
instruction:'{nl}'
Response:
'''
    start_time = time.time()

    completion = client.chat.completions.create(
    model="deepseek-v3-2-exp", 
    messages=[
        {"role": "user", "content": PROMPT+prompt_nl}
    ],
    #stream=False    
    )
    
    res = completion.choices[0].message.content


    
    end_time = time.time()
    used_time = end_time - start_time
    #print('llm生成：',res)
    print('llm的调用时间为：',used_time)
    return res, used_time



if __name__ == "__main__":
    # 读取数据
    input_file = '/home/lijia/code/1113_CLHS/train_dataset/1208_dataset_v2/negative_robustness.json'#'/home/lijia/code/LLaMA-Factory/data/household_2450_cot_nl.json'#'/home/lijia/code/1113_CLHS/train_dataset/train_stl_json_demo_2_1203.json'
    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    '''# 随机抽取10个dict单元
    sampled_items = random.sample(data, 10)
    results = []

    for idx, item in enumerate(sampled_items): #sampled_items
        instruction = item['instruction']
        print(f"处理第 {idx+1} 条指令: {instruction}")
        stl_json, used_time = call_llm(instruction)

        print(f"生成的STL JSON: {stl_json}")
        results.append({
            'instruction': instruction,
            'stl_json': stl_json,
            'used_time': used_time
        })

    # 保存到新文件
    output_file = '/home/lijia/code/1113_CLHS/train_dataset/train_stl_json_demo_2_1204.json'
    with open(output_file, 'w', encoding='utf-8') as f_out:
        import json
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print(f"已保存到: {output_file}")'''

    # 依次处理每个dict单元
    results = []
    batch_size = 100  # 每100条保存一次
    for idx, item in enumerate(data):
        instruction = item['instruction']
        print(f"处理第 {idx+1} 条指令: {instruction}")
        stl_json, used_time = call_llm(instruction)
        print(f"生成的STL JSON: {stl_json}")
        
        results.append({
            'instruction': instruction,
            'stl_json': stl_json,
            'used_time': used_time
        })
        
        # 每处理batch_size条或最后一条时保存
        if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
            output_file = '/home/lijia/code/1113_CLHS/train_dataset/1208_dataset_v2/negative_robustness_stljson.json'
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, ensure_ascii=False, indent=2)
            print(f"已保存前 {len(results)} 条结果到: {output_file}")
    
    print(f"处理完成，共 {len(results)} 条结果。")

