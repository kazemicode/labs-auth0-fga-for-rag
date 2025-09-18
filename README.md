![img](https://cdn.auth0.com/website/auth0-training/auth0-labs-ext/oktane24.png)

# Lab: Fine-grained Authorization for RAG with Auth for AI Agents

FGA+RAG? OMG! Leverage Auth0 FGA and Auth for AI Agents to implement fine-grained authorization for Retrieval Augmented Generation (RAG). When a user prompts an AI agent for for information, the AI agent uses the RAG pipeline to generate a response. But what if the response has sensitive data? Using FGA, you can ensure the user has authorization to receive the data in the response before the response is rendered.

This lab uses adapted sample code from Auth0 documentation on how to set up [Authorization for RAG](https://auth0.com/ai/docs/authorization-for-rag) using Auth0 FGA.


---
# How to Get Started with this Lab

## Create Required Free Accounts

1. A GitHub account with Codespaces access (you can use the free plan, which gives you 60 hours of use per month). Create a free account [here](https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account) if you do not already have one.

    **⚠️ Note to internal Okta employees: Do NOT use your EMU (Enterprise-managed user) account since Codespaces are disabled.**

2. An Auth0 FGA account: You'll need an Auth0 FGA account. We'll walk you through how to set this up in the lab.
3. A Google Gemini: Last, you'll need an Google Gemini account to create an Google Gemini API key for this lab. We'll walk you through how to set this up in the lab. While in practice you can use any model provider, we use Google Gemini since free-tier users can generate embeddings.

## Accessing and Working with Auth0 Labs

1. **After logging into GitHub, open the lab repo in a Codespace:** From the *Code* dropdown menu of this branch, toggle to the *Codespaces* tab. Click the plus sign to create and open the lab in a Codespace. A new tab will open, and Codespaces will begin configuring the lab environment. Wait for the environment to finish building.
2. **Begin working with the lab:** Once the environment loads all the required libraries and extensions, you'll see a Codetour popup with instructions on how to get started. At this point, you should follow the remainder of the instructions within Codetour!

### Notes:
- **If you'd like to save your work to your own fork:** You can commit and push your changes to a fork of this repo. (See: [Using Source Control in Your Codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/using-source-control-in-your-codespace)).
- **Close the Codespace when you're finished with the lab:** Codespaces come with a set amount of free usage. To avoid using all of your free use allocation, be sure to return to the forked repo, select the "Code" dropdown, select the dots next to your open Codespace, and select "Delete." This will not delete your forked repository. You can keep that forever, and open a new Codespace whenever you like.
- Questions? Check out our [Lab FAQs](https://auth0-training.github.io/)!


### Issue Reporting
---
If you have found a bug or if you have a feature request, please report them at this repository issues section. Please do not report security vulnerabilities on the public GitHub issue tracker. The [Responsible Disclosure Program](https://auth0.com/whitehat) details the procedure for disclosing security issues.

### Author
---

[Auth0](https://auth0.com)

### License
---

This project is licensed under the MIT license. See the [LICENSE](LICENSE.txt) file for more info.
