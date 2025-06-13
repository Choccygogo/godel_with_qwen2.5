from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 Goedel-Prover 模型和专用分词器
model_name = "Goedel-LM/Goedel-Prover-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

problem_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/- prove that $a^{3}+b^{3}+c^{3}+{\frac {15\,abc}{4}} \geq \frac{1}{4}$ given $a, b, c,$ are non-negative reals such that $a+b+c=1$ -/
theorem lean_workbook_10009 (a b c: ℝ) (ha : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1): a^3 + b^3 + c^3 + (15 * a * b * c)/4 ≥ 1/4 := by sorry
"""

inputs = tokenizer(problem_statement, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,  # 控制证明长度
    num_return_sequences=1,  # 生成多个候选证明
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# 解码输出
proofs = [tokenizer.decode(out, skip_special_tokens=True)
          for out in outputs]

# Save proofs to a file
with open('generated_proofs.txt', 'w', encoding='utf-8') as f:
    f.write("Generated Proofs:\n")
    f.write("=" * 50 + "\n\n")
    for i, proof in enumerate(proofs, 1):
        f.write(f"Proof {i}:\n")
        f.write(proof)
        f.write("\n\n" + "=" * 50 + "\n\n")

print(f"Proofs have been saved to 'generated_proofs.txt'")