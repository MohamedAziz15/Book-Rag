from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def retrieve_answer(question: str) -> str:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI


    retriever = ...  # Your retriever
    llm = ChatOpenAI()

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain.invoke({"input": question})["output"]
    


if __name__ == "__main__":
    response = gemini_llm.invoke(["What is the capital of France?"])
    print(response)

