# From Data to Translation: Leveraging AI for Efficient and Accurate Translation of Scientific Reports 

# Phase 5: Final AI Translation Model and Translation Quality Evaluation

## Description

This phase combines the fine-tuned translation models from Phase 2 with the rule-based preferential translation system from Phase 3 to create the final integrated translation solution. The combined system leverages both the contextual understanding of fine-tuned models and the precision of rule-based terminology preservation to deliver high-quality scientific translations.

A comprehensive evaluation framework has been established to assess translation quality through both analytical metrics and human evaluation surveys. This phase focuses on optimising the integration between neural and rule-based components while preparing for systematic quality assessment through blind survey evaluation.

## Key Components

### Integrated Translation System
- Combined fine-tuned translation models with rule-based terminology preservation
- Optimised integration parameters between neural and rule-based components
- End-to-end translation pipeline for CSAS scientific documents
- Performance monitoring and quality metrics

### Translation Quality Evaluation Framework
- Analytical evaluation using similarity metrics to previously published translations
- Integration with Phase 4 survey application for blind human evaluation
- Comparative analysis across different model configurations
- Statistical validation of translation improvements

### Model Selection and Optimisation
- Performance comparison across multiple fine-tuned model variants
- Rule-based layer effectiveness evaluation
- Hyperparameter optimisation for the integrated system
- Final model selection based on comprehensive evaluation criteria

### Best-of-Ensemble Combined Model
- In addition the individual translation models, another Best-of-Ensemble model is created by combining results from all other models
- For each sentence, the highest similarity between source and translated text (after excluding errors) is considered to be the Best-of-Ensemble or the "ensemble model"
- The ensemble model returns improved translations, while requiring more time and computational resources to complete translations

## Evaluation Methodology

### Analytical Assessment
- Translation quality metrics and similarity calculations
- Comparison with baseline translation models
- Performance evaluation against previously published CSAS translations
- Technical accuracy assessment for scientific terminology

### Human Evaluation (In-Progress)
- Blind survey evaluation using randomised translation samples
- Quality ratings from domain experts and translation professionals
- Comparative ranking between different model configurations
- Statistical analysis of human evaluation results

## Current Status

The integrated translation system has been successfully implemented and analytical evaluation is complete. Human evaluation through the survey application is currently in progress, with results pending to inform final model selection and deployment preparation.

## Additional Notes About Energy Consumption

With the inclusion of the ensemble model described above, energy consumption was flagged as a potential concern. The ensemble model could increase energy and costs by as much a factor of 10, since all translations are performed for each translation model, including fine-tuned and base models. However, analysis indicates that ensemble model translations would likely utilise less than 16 kWhr of energy over the course a year (assuming a conservatively high 200 documents are translated). This energy usage would lead to approximately $1.56 in estimated yearly electrical costs. The energy impacts have therefore been deemed to be negligible for this project.

Important to note, if a similar project utilises this approach and has a much larger volume of translations, energy and costs may become a factor. In this case it is worth noting that costs (and time) could be reduced by an order of magnitude if the smallest translation model is utilised, instead of the ensemble model. Additional energy reductions could be obtained by further optimising the deployed system configuration and/or eliminating repeated attempts to deal with token replacement errors (as described in Phase 3).

## Expected Outcomes

Upon completion of the survey evaluation, this phase will deliver:
- Validated final translation model optimised for CSAS scientific documents
- Comprehensive quality assessment comparing multiple approaches
- Performance benchmarks against existing translation solutions
- Deployment-ready integrated translation system

## All Phases

- **Phase 1**: [Data Gathering and Transformation](https://github.com/KevinCarr42/AI-Translation) (complete)
- **Phase 2**: [AI Translation Fine-Tuning](https://github.com/KevinCarr42/Translation-Fine-Tuning) (complete)
- **Phase 3**: [Rule-Based Preferential Translations](https://github.com/KevinCarr42/rule-based-translation) (complete)
- **Phase 4**: [AI Translation Quality Survey App](https://github.com/KevinCarr42/translation-quality-survey-app) (complete)
- **Phase 5**: Final AI Translation Model and Translation Quality Evaluation (in-progress)
- **Phase 6**: Deploy the Final Model (in-progress)
