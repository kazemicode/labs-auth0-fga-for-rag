import fs from "node:fs/promises";
import { StructuredToolInterface } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { Document } from "@langchain/core/documents";

export class RetrievalAgent {
  private agent;

  private constructor(agent: any) {
    this.agent = agent;
  }

  // Create a retrieval agent with a retriever tool and a language model
  static create(tools: StructuredToolInterface[]) {
    // Create a retrieval agent that has access to the retrieval tool.
    const retrievalAgent = createReactAgent({
      llm: new ChatOpenAI({ temperature: 0, model: "gpt-4o-mini" }),
      tools,
      stateModifier: [
        "Answer the user's question only based on context retrieved from provided tools.",
        "Only use the information provided by the tools.",
        "If you need more information, ask for it.",
      ].join(" "),
    });

    return new RetrievalAgent(retrievalAgent);
  }

  // Query the retrieval agent with a user question
  async query(query: string) {
    const { messages } = await this.agent.invoke({
      messages: [
        {
          role: "user",
          content: query,
        },
      ],
    });

    return messages.at(-1)?.content;
  }
}

async function readDoc(path: string) {
  return await fs.readFile(path, "utf-8");
}

/* Reads documents from the assets folder and converts them to langChain Documents */
export async function readDocuments() {
  const folderPath = "./assets";
  const files = await fs.readdir(folderPath);
  const documents: Document[] = [];

  for (const file of files) {
    documents.push(
      new Document({
        pageContent: await readDoc(`${folderPath}/${file}`),
        metadata: { id: file.slice(0, file.lastIndexOf(".")) },
      })
    );
  }

  return documents;
}
