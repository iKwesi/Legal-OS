# 1. Goals and Background Context

## 1.1 Goals

* **Primary Goal:** Build an end-to-end prototype that satisfies all requirements of the **AIE8 Certification Challenge Rubric**.
* **Rubric-Driven Goals:**
    * Deliver a `localhost` deployable full-stack application.
    * Generate a **Synthetic Golden Dataset (SGD)** using the **RAGAS** framework's testset generation capabilities. This SGD will be the official benchmark for all subsequent RAGAS evaluations of the pipeline.
    * Assess the RAG pipeline using RAGAS for faithfulness, response relevance, context precision, and context recall.
    * Quantify performance improvements from swapping a base retriever with an advanced one, providing results in a table.
    * Deliver a 5-minute Loom demo script and all project code/documentation.
* **Product Goals:**
    * Successfully implement the full "Transactional Cluster" as a modular, six-agent system (Ingestion, Clause Extraction, Risk Scoring, Summary, Checklist, Provenance) using the ReAct pattern where appropriate.
    * Develop a Jupyter Notebook environment (using Jupytext-compatible Python scripts) to independently test each agent and their collaboration.
    * Architect the system for future expansion into the full "Legal OS" (e.g., multi-jurisdictional support).

## 1.2 Background Context

This project represents the foundational "Transactional Cluster" module of a larger "Legal OS" vision. The immediate problem we are solving is the slow, manual, and error-prone nature of M&A due diligence, a critical, non-billable risk for legal teams.

Our solution is a multi-agent RAG application that automates the entire diligence workflow: ingesting documents, extracting clauses, scoring risk, and generating auditable summary reports.

For this AIE8 Certification Challenge, the prototype will be a full-stack `localhost` application built using publicly available U.S. contract datasets (like SEC EDGAR, specific dataset TBD by Architect) as the initial data source. The primary deliverables are not just the working application, but also the verifiable test results from RAGAS, the demonstration of a modular, collaborative multi-agent architecture (using ReAct and LangGraph), and clear documentation addressing the rubric.

## 1.3 Change Log

| Date   | Version | Description                          | Author    |
| :----- | :------ | :----------------------------------- | :-------- |
| (Auto) | 1.0     | Initial PRD draft based on brief | John (PM) |
| (Auto) | 1.1     | Added SGD/RAGAS/Retriever/Notebook details, refined UI goals, added Chat, re-added Video Script | John (PM) |

---
