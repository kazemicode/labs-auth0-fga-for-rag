import "dotenv/config";

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FGARetriever } from "@auth0/ai-langchain/RAG";

import { readDocuments, RetrievalAgent } from "./helpers/langchain";

async function main() {
  console.info(
    "\n..:: LangGraph Agents Example: Agentic Retrieval with Auth0 FGA \n\n"
  );

  // 0. Define a user (String)

  // 1. Read and load documents from the assets folder and store as List of Documents

  // 2. Create an in-memory vector store (MemoryVectorStore) of embeddings for OpenAI models from the list of Documents.

  // 3. Create a retriever that uses Auth0 FGA to filter documents based on user permissions.

  // 4. Convert the retriever into a tool for an agent.

  // 5. Create the retrieval agent

  // 6. Query the retrieval agent with a prompt
  // The agent will rephrase the original question and populate the "query" argument, until it can answer the user's question.

// 7. Print the answer

}

main().catch(console.error);
