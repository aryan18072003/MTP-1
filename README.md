# MTP-1

BioHopR evaluation using biomedical knowledge graphs and MedGemma.

- [Knowledge graph evaluation](src/kg_eval/README.md): PrimeKG, Hetionet, DRKG and BioKG retrieval.
- [Standalone MedGemma baseline](src/medgemma_baseline/README.md): model-only BioHopR evaluation, separate preparation and results, with no KG downloads.
- [MedGemma evaluation](src/medgemma_eval/README.md): MedGemma 27B alone and with each graph, scored on the same questions.
- [Remote GPU commands](src/medgemma_eval/RUN_REMOTE.md): environment setup, smoke run, full run and comparison on the H100 server.
