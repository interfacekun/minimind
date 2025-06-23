
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer():
    # https://yuanbao.tencent.com/favorites/cfd69845-2349-4d15-8d90-2efcbe6f5524
    # https://yuanbao.tencent.com/favorites/c25f01a4-c8b9-4c5e-bff0-6aaddb2efce1
    # 初始化 BPE 模型
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 使用空格预分词
    tokenizer.pre_tokenizer = Whitespace()

    # 配置训练器
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=50  # 小词汇表便于演示
    )

    # 训练 tokenizer
    tokenizer.train(files=["example.txt"], trainer=trainer)
    return tokenizer

def test(tokenizer):
    vocab = tokenizer.get_vocab()
    print("词汇表大小:", len(vocab))
    for token, idx in sorted(vocab.items(), key=lambda x: x[1])[:50]:  # 打印前50个
        print(f"{idx:5d}: {token}")
    # 编码新文本
    output1 = tokenizer.encode("Deep learning is awesome!")
    output2 = tokenizer.encode("Have you used BERT?")

    print("第一个句子分词结果:")
    print(f"Tokens: {output1.tokens}")
    print(f"IDs: {output1.ids}")
    print(f"Attention mask: {output1.attention_mask}")

    print("\n第二个句子分词结果:")
    print(f"Tokens: {output2.tokens}")
    print(f"IDs: {output2.ids}")
    print(f"Attention mask: {output2.attention_mask}")

def main():
    # 训练 tokenizer
    tokenizer = train_tokenizer()

    # 测试 tokenizer
    test(tokenizer)

    # 保存 tokenizer
    tokenizer.save("tokenizer.json")

if __name__ == "__main__":
    main()
