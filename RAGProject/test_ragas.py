import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

load_dotenv()

# 沿用你项目的配置
from config import key, url, embed_model_ch


async def main():
    # 1. 加载本地 Embedding 模型
    # 注意：这是 Ragas 的类，参数叫 model，不是 model_name
    embeddings = HuggingFaceEmbeddings(
        model=str(embed_model_ch),
        device="cpu",
    )

    # 2. 连接评分用的 LLM
    async with AsyncOpenAI(
        api_key=key,
        base_url=url,
    ) as client:

        judge_llm = llm_factory(
            "deepseek-flash",
            client=client,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

        # 3. 创建两个评估指标
        faithfulness = Faithfulness(
            llm=judge_llm,
        )

        relevancy = AnswerRelevancy(
            llm=judge_llm,
            embeddings=embeddings,
        )

        # 4. 准备一条待评估的数据
        question = "小明几岁，住在哪里？"

        answer = "小明今年20岁，住在上海。"

        contexts = [
            "小明今年20岁，目前住在北京。"
        ]

        # 5. 评估忠实度：回答是否有上下文依据
        faithfulness_result = await faithfulness.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        # 6. 评估相关性：回答是否切题
        relevancy_result = await relevancy.ascore(
            user_input=question,
            response=answer,
        )

        # 7. 输出分数
        print("忠实度：", faithfulness_result.value)
        print("回答相关性：", relevancy_result.value)


if __name__ == "__main__":
    asyncio.run(main())