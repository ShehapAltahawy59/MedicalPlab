import json
import random
def create_data_file():
    with open('data/Doctor_Trainer.questions.json', 'r') as f:
        data = json.load(f)

        formatted_data = []

        for item in data:
            # Prepare options and their original labels
            options = [
                ("A", item['choice_a']),
                ("B", item['choice_b']),
                ("C", item['choice_c']),
                ("D", item['choice_d']),
                ("E", item['choice_e'])
            ]
            # Shuffle options
            random.shuffle(options)
            # Build output string
            output = f"{item['question']}\n\n"
            for idx, (label, text) in enumerate(options):
                output += f"{chr(ord('A') + idx)}. {text}\n"
            output += f"\nAnswer: {item['answer']}\n\n"
            formatted_data.append({
                "instruction": f"instruction: Write a multiple-choice medical question with 5 options and the correct answer on {item['topic']}. Include exactly five answer options (A–E), and specify the correct one at the end like 'Answer: '.",
                "input": "",
                "output": output
            })

        with open('Formatted data/qlora_ready_data.json', 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')  # JSONL format

create_data_file()
