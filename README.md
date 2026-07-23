# Predicting Customer Purchase Behavior with Luma Event Data

Link to notebook: https://github.com/ishaans799-lab/Initial-Report-and-Exploratory-Data-Analysis-EDA-/blob/206a52efd21df50b553f6eed838bfee0584a608e/luma_customer_purchase_eda.ipynb

## Executive Summary

Modern digital businesses interact with customers across web, mobile, and email touchpoints, but it is difficult to know which customers are most likely to purchase and what engagement action should happen next. This project uses the Luma event dataset to explore whether machine learning can predict customer purchase behavior and support personalized next-best-action recommendations.

The initial analysis cleaned and aggregated event-level activity into customer-level journeys, engineered behavioral features, explored conversion patterns, created unsupervised customer segments, and trained a baseline supervised classification model. The baseline logistic regression model substantially outperformed a majority-class dummy model, suggesting that behavioral signals in the Luma data can be used to predict purchase propensity.

## Rationale

Customer engagement decisions are often made using broad campaign rules, such as sending the same email or offer to every customer. A machine learning approach can make these decisions more targeted by identifying customers who show high purchase intent, customers who need nurturing, and customers who are unlikely to convert without a different strategy. Better targeting can improve conversion rates, reduce irrelevant messaging, and increase long-term customer value.

## Research Question

Can machine learning be used to predict customer purchase behavior and recommend personalized engagement actions that improve conversion outcomes?

## Data Sources

This project uses `luma_post_extended.csv`, an event-level Luma customer interaction dataset containing:

- Web activity, including page views, site sections, referrers, link clicks, and product interactions
- Commerce events, including product views, cart adds, checkouts, purchases, order IDs, and parsed order value
- Marketing information, including email campaign tracking codes
- Customer and device context, including customer IDs, country, device/browser fields, and app activity

The raw file contains 46,110 rows and 95 columns. After removing rows without a usable customer ID, the working event table contains 46,108 events across 929 customers.

## Methodology

The analysis is organized around three modeling tasks:

1. **Exploratory data analysis and cleaning**
   - Converted timestamps and sparse event counters into usable formats
   - Removed duplicate rows and rows without customer IDs
   - Treated missing event counter fields as zero because they represent sparse event indicators
   - Parsed revenue from `productListItems` where available
   - Checked missingness, conversion distribution, outliers, referrer patterns, and email engagement

2. **Feature engineering**
   - Aggregated event-level data into one customer-level record per customer
   - Created features for event depth, sessions, page views, product views, cart adds, checkouts, unique pages, unique site sections, email touches, mobile app activity, journey duration, cart-add rate, and checkout rate
   - Used only pre-purchase events for converted customers when building model features to reduce target leakage

3. **Machine learning**
   - Built a baseline logistic regression classifier to predict whether a customer converts
   - Compared the classifier against a dummy majority-class baseline
   - Evaluated accuracy, precision, recall, F1 score, and ROC-AUC
   - Used K-means clustering to identify behavioral customer segments
   - Created a rule-based next-best-action prototype using predicted propensity and segment behavior

## Results

The cleaned customer-level dataset contains 929 customers. Of these, 224 customers converted, producing a customer-level conversion rate of approximately 24.1%. The dataset includes 317 unique orders and approximately $63,798 in parsed revenue.

EDA showed several strong behavioral patterns:

- Customers with an email campaign touch converted at about 64.2%, compared with about 12.0% for customers without an email touch.
- Internal referrer/navigation behavior was associated with much higher conversion than typed/bookmarked, search, external, or social referrers.
- Converters generally had deeper journeys, more product engagement, more cart activity, and more checkout activity than non-converters.
- Outlier analysis showed a small number of customers with extremely high event counts and product interactions, which should be monitored in future modeling.

The baseline logistic regression model strongly outperformed the dummy baseline:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy majority baseline | 0.760 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression baseline | 0.996 | 1.000 | 0.982 | 0.991 | 0.994 |

Permutation importance identified checkout activity, checkout rate, cart adds, unique pages, and primary referrer as the strongest predictors in the baseline model. These findings are intuitive: customers who reached checkout or added items to cart showed clear purchase intent.

K-means clustering produced interpretable behavioral groups, including:

- A large low-intent segment with low conversion and little checkout activity
- A high-intent segment with high cart/checkout activity, high email exposure, and high revenue
- Smaller heavy-browsing or app-heavy segments that may need more investigation before production use

The initial next-best-action prototype maps customers into engagement actions such as checkout recovery, high-intent product recommendations, email-nurture offers, and low-intent awareness content.

## Interpretation and Limitations

The analysis supports the research question: machine learning can predict customer purchase behavior from Luma event data and can be connected to next-best-action decisioning logic. However, the baseline model is likely optimistic because cart and checkout features are very close to the purchase event. The current model is best interpreted as an initial benchmark for Module 20 rather than a final production model.

For Module 24, the model should be tested with stricter prediction windows, such as predicting purchase before cart add or before checkout. Future models should also compare tree-based classifiers, tune decision thresholds based on business costs, and validate recommendations with holdout data or A/B testing.

## Next Steps

- Create time-based prediction windows to reduce leakage and better simulate real-time marketing decisions.
- Compare logistic regression with Random Forest, Gradient Boosting, and other classification models.
- Tune classification thresholds for different business objectives, such as maximizing recall for high-value customers or precision for discount offers.
- Add customer lifetime value or order value prediction as a second supervised learning task.
- Refine customer segments with additional clustering methods and business labels.
- Test next-best-action rules using campaign response or experimental data.


## Contact and Further Information

This repository contains the initial capstone EDA report and baseline model notebook for the Luma customer purchase propensity project.
