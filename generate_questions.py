from transformers import pipeline
from fine_tuning import get_model
# Create pipeline


def generate_anatomy_question(prompt):
        
        output = generator(prompt, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8)
        return output[0]['generated_text']


if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    model,tokenizer = get_model(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "instruction: Write a multiple-choice medical question with 5 options and the correct answer on ANATOMY."
    # Generate and save questions
    with open("output_questions/anatomy_questions.txt", "w") as f:
        for i in range(10):
            try:
                question = generate_anatomy_question(prompt)
                # Clean up the output if needed
                question = question.replace(prompt, "").strip()
                f.write(f"Question {i+1}:\n{question}\n\n")
                print(f"Generated question {i+1}")
                
                # Add a small delay to avoid overloading
                
                
            except Exception as e:
                print(f"Error generating question {i+1}: {str(e)}")
                continue

    print("Finished generating 10 anatomy questions!")
