                           -Random Forest-
The Random Forest classifier achieved an accuracy of approximately 0.84,
demonstrating strong predictive performance on the HR Employee Attrition dataset. 
Considering the class imbalance present in the data, additional evaluation metrics 
such as precision, recall, and F1-score were used to provide a more reliable assessment
of the model. The results show that the model is effective in identifying employees at
risk of attrition while maintaining good generalization without overfitting. Overall,
the Random Forest approach significantly outperformed the rule-based system and proved
suitable for this prediction task.

The Random Forest model achieved an accuracy of 84.35%, 
which reflects the class imbalance present in the dataset, 
where the majority of employees do not leave the company. 
Although overall accuracy is high, the recall for attrition 
cases is low, indicating that the model is conservative in predicting 
employee turnover. In contrast, the rule-based system demonstrates high
precision and recall but applies only to a small subset of employees (14%), 
prioritizing confident decisions. The hybrid sequential approach combines both 
methods, improving recall over the Random Forest while maintaining full coverage. 
Feature importance analysis shows that income, age, overtime, and tenure are 
the most influential factors in predicting employee attrition.