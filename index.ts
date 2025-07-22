import "dotenv/config";

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FGARetriever } from "@auth0/ai-langchain/RAG";

import { readDocuments, RetrievalAgent } from "./helpers/langchain";

async function main() {
  console.info(
    "\n..:: LangGraph Agents Example: Agentic Retrieval with Auth0 FGA \n\n"
  );

  const user = "user1";
  // 1. Read and load documents from the assets folder
  const documents = await readDocuments();
  // 2. Create an in-memory vector store from the documents for OpenAI models.
  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    new OpenAIEmbeddings({ model: "text-embedding-3-small" })
  );
  // 3. Create a retriever that uses FGA to gate fetching documents on permissions.
  const retriever = FGARetriever.create({
    retriever: vectorStore.asRetriever(),
    // FGA tuple to query for the user's permissions
    buildQuery: (doc) => ({
      user: `user:${user}`,
      object: `doc:${doc.metadata.id}`,
      relation: "viewer",
    }),
  });
  // 4. Convert the retriever into a tool for an agent.
  const fgaTool = retriever.asJoinedStringTool();
  // 5. The agent will call the tool, rephrasing the original question and
  // populating the "query" argument, until it can answer the user's question.
  const retrievalAgent = RetrievalAgent.create([fgaTool]);
  // 6. Query the retrieval agent with a prompt
  const answer = await retrievalAgent.query("Show me forecast for ZEKO?");

  console.info(answer);
}

main().catch(console.error);
