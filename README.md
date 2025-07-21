# Medical Question Generation and Fine-Tuning

This project provides a workflow for preparing data, fine-tuning a large language model (LLM), and generating new multiple-choice medical questions. It leverages HuggingFace Transformers, PEFT, and TRL libraries for efficient model training and inference.

## Project Structure

- `functions/create_data_file.py`: Prepares and formats raw medical question data for model training.
- `fine_tuning.py`: Fine-tunes a language model using the prepared data.
- `generate_qustion.py`: Generates new medical questions using the fine-tuned model.
- `data/Doctor_Trainer.questions.json`: Source data of medical questions (input).
- `Formatted_data/qlora_ready_data.json`: Formatted data for training (output of `create_data_file.py`).
- `output_questions/`: Directory for generated questions.
- `weights/`: Directory for model checkpoints and weights.

## Setup

1. **Clone the repository and install dependencies:**

```bash
pip install -r requirements.txt
```

Dependencies include:
- `transformers`
- `datasets`
- `trl`
- `peft`
- `huggingface_hub`

2. **Prepare your HuggingFace API key:**
   - Update `fine_tuning.py` with your HuggingFace API key in the `login()` function.

3. **Ensure your data file is present:**
   - Place your raw questions file at `data/Doctor_Trainer.questions.json`.

## Usage

### 1. Prepare Data

Run the following to convert your raw data into the format required for fine-tuning:

```bash
python create_data_file.py
```

This will create `qlora_ready_data.json` in the project root.

### 2. Fine-Tune the Model

Run the fine-tuning script:

```bash
python fine_tuning.py
```

This will:
- Prepare the model with LoRA and quantization for efficient training.
- Load and shuffle the formatted dataset.
- Fine-tune the model and save checkpoints to the `weights/` directory.

### 3. Generate New Questions

After fine-tuning, generate new questions (e.g., on anatomy):

```bash
python generate_qustion.py
```

This will use the fine-tuned model to generate 10 new questions and save them to `output_questions/anatomy_questions.txt`.

## Notes
- You can modify the prompt in `generate_qustion.py` to generate questions on different topics.
- Make sure you have enough system resources (RAM/GPU) for model training and inference.

## License
MIT 



# ✅ Medical Question Evaluation Report

## 📚 Evaluation Criteria:

Each question is evaluated using the following criteria:

- **Related to Topic?** – Is the question relevant to the intended subject (Anatomy or Pharmacology)?  
- **Medical Question?** – Is the question medically oriented?  
- **One Correct Option?** – Is there a single clearly correct answer?  
- **Correct Option Present?** – Is the correct answer among the listed options?  
- **Medically Accurate?** – Is the correct answer scientifically and medically valid?

---

## 🩻 Anatomy Questions Evaluation

| Q# | Related to Anatomy? | Medical Question? | One Correct Option? | Correct Option Present? | Medically Accurate? | Notes |
|----|----------------------|--------------------|----------------------|--------------------------|----------------------|-------|
| 1  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | ACL stabilizes the knee. |
| 2  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Muscles produce movement. |
| 3  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ❌ No               | Scalenus medius is not between clavicles and sternum. |
| 4  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Occipital bone is correct. |
| 5  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | SVC is anatomically above the aorta. |
| 6  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ❌ No               | "Myofascial" is not a muscle; correct could be digastric. |
| 7  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Sphenopalatine ganglion helps regulate breathing. |
| 8  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Ventricles are on right side of heart. |
| 9  | ✅ Yes              | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Rib is a human bone. |
| 10 | ✅ Yes                | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Clinical diagnosis; not anatomy-focused. |

### ✅ Anatomy Summary:

- **Related to Anatomy**: 9/10  
- **Medical Questions**: 10/10  
- **One Correct Option**: 10/10  
- **Correct Option Present**: 10/10  
- **Medically Accurate Answers**: 8/10  

---

## 💊 Pharmacology Questions Evaluation

| Q# | Related to Pharmacology? | Medical Question? | One Correct Option? | Correct Option Present? | Medically Accurate? | Notes |
|----|---------------------------|--------------------|----------------------|--------------------------|----------------------|-------|
| 1  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ❌ No               | 80 units insulin for 150 mg/dL is dangerously high. |
| 2  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Calcium channel blockers treat hypertension. |
| 3  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | SSRI + symptoms suggest serotonin syndrome. |
| 4  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ❌ No               | Ampicillin contraindicated in penicillin allergy. |
| 5  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Activated charcoal correct for overdose. |
| 6  | ✅ Yes                     | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | COPD correct, but diagnosis-based, not pharmacology. |
| 7  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Constipation is a common anticholinergic effect. |
| 8  | ✅ Yes                     | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Clinical case, not pharmacology-focused. |
| 9  | ✅ Yes                   | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ✅ Yes              | Statins reduce cardiovascular risk. |
| 10 | ✅ Yes                     | ✅ Yes            | ✅ Yes              | ✅ Yes                  | ❌ No               | Diagnosis unclear; not pharmacologically focused. |

### ✅ Pharmacology Summary:

- **Related to Pharmacology**: 10/10  
- **Medical Questions**: 10/10  
- **One Correct Option**: 10/10  
- **Correct Option Present**: 10/10  
- **Medically Accurate Answers**: 7/10  
