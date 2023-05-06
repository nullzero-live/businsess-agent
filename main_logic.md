# How the agent conversation will occur

1. The consultant agent will ask the Startup Founder what their idea is. This will be stored as a variable called `idea`.
2. The consultant agent will ask the Startup Founder what their target market is. This will be stored as a variable called `target_market`.
3. The consultant agent will ask the Startup Founder what their business model is. This will be stored as a variable called `business_model`.
4. The consultant agent will ask the Startup Founder what their revenue model is. This will be stored as a variable called `revenue_model`.

Notes:
- The Startup Founder will be able to answer the questions in one order.
- Each step will evolve the conversation from the perspective of the consultant.
- The startup and consultant need to be prompted to only ask individual questions when appropriate.
- The result will converge on each specific variable name in order to move through the logic flow to the next level.