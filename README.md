<!-- Title -->
<head>
    <h1 align='center'><b><u><i>
        Telco Project - README
    </i></u></b></h1>
</head>





<!-- Table of Contents -->
<head>
    <h3 align='center'><b><i>
        <a id='tableofcontents'></a>Table of Contents:
    </i></b></h3>
</head>
<h5>
<li><a href='#description'>Project Description</a></li>
<li><a href='#goals'>Project Goals</a></li>
<li><a href='#hypo'>Hypothesis/Questions</a></li>
<li><a href='#datadict'>Data Dictionary</a></li>
<li><a href='#planning'>Planning</a></li>
<li><a href='#instructions'>Instruction To Replicate</a></li>
<li><a href='#takeaways'>Takeaways</a></li>
</h5>
<br><br><br>




<!-- Project Description -->
<head>
    <h3 align='center'><b><i>
        <a id='description'></a>Project Description:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h5>
Using the 'telco' dataset from a SQL database, figure out what are causes to customer churn within the company.
</h5>
<br><br><br>





<!-- Project Goals -->
<head>
    <h3 align='center'><b><i>
        <a id='goals'></a>Project Goals:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h5>
<li>Implement Data Science Pipeline</li>
<li>Acquire 'Telco' Dataset</li>
<li>Prepare 'Telco' Dataset</li>
<li>Explore 'Telco' Dataset</li>
<li>Model 'Telco' Dataset</li>
<li>Deliver 'Telco' Dataset</li>
</h5>
<br><br><br>





<!-- Hypothesis/Questions -->
<head>
    <h3 align='center'><b><i>
        <a id='hypo'></a>Hypothesis/Questions:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>
Hypothesis:
</b></h4>
<h5>
Given that 'Telco' is a company providing a cellular and internet service to consumers(People), I believe that being able to identify what is deemed as valuable to the consumer weighed by it's cost is the primary cause to customer churn.
</h5>
<br>
<h4><b>
Questions:
</b></h4>
<h5>
<li>What does telco offer</li>
<li>What is the price of everything</li>
<li>Are there specific services that retain better</li>
<li>Are there specific services that retain worse</li>
<li>Are there demographics of individuals</li>
<li>What sort of advertisements are posted</li>
<li>Are there customer comments for those who churned</li>
<li>Are there customer comments for those who joined</li>
<li>Are there timestamps for joining and leaving</li>
<li>Are there customer satisfaction reports</li>
<li>Are there reports of faulty services/equipment</li>
<li>What is the situation locally(placement, access, patterns, crime, common knowledge/beliefs/morals/ethics of community)</li>
</h5>
<br><br><br>






<!-- Data Dictionary -->
<head>
    <h3 align='center'><b><i>
        <a id='datadict'></a>Data Dictionary:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>

| Field Name | Data Type | Data Format | Description | Example |
| ----- | ----- | ----- | ----- | ----- |
| gender | object | str | Defines gender of customer | 'Male' |
| senior_citizen | int | 1, 0 | Defines customer as being of senior-citizen status or not | 1 |
| partner | object | str | Defines customer as having a partner or not | 'Yes' |
| dependents | object | str | Defines customer as having dependents or not | 'Yes' |
| tenure | int | # | Defines customer's tenure in months | 12 |
| phone_service | object | str | Defines customer as having phone service or not | 'Yes' |
| multiple_lines | object | str | Defines customer as having multiple lines or not | 'Yes' |
| online_security | object | str | Defines customer as having online security or not | 'Yes' |
| online_backup | object | str | Defines customer as having online backup or not | 'Yes' |
| device_protection | object | str | Defines customer as having device protection or not | 'Yes' |
| tech_support | object | str | Defines customer as having tech support or not | 'Yes' |
| streaming_tv | object | str | Defines customer as having tv streaming service or not | 'Yes' |
| streaming_movies | object | str | Defines customer as having movie streaming service or not | 'Yes' |
| paperless_billing | object | str | Defines customer as having paperless billing or not | 'Yes' |
| monthly_charges | float | 00.00 | Defines customer's monthly charges in USD($) | 37.48 |
| total_charges | float | 00.00 | Defines customer's total charges with telco company in USD($) | 7862.44 |
| churn | object | str | Defines customer churn status within the telco company | 'Yes' |
| churn_month | datetime | YYYY-MM-DD | Defines customer's date of churn | 2022-01-31 |
| signup_date | datetime | YYYY-MM-DD | Defines customer's date of signing-up with the company | 2021-04-21 |
| contract_type | object | str | Defines customer's contract type | 'Month-to-month' |
| internet_service_type | object | str | Defines customer's internet service type | 'DSL' |
| payment_type | object | str | Defines customer's payment type | 'Mailed Check' |
| DUMMY COLUMNS | uint | 0, 1 | Dummy columns generated from object columns | 0 |
<br><br><br>






<!-- Planning -->
<head>
    <h3 align='center'><b><i>
        <a id='planning'></a>Planning:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>Planning</b></h4>
<li>This good ol’ thingy</li>
<br>
<h4><b>Acquire</b></h4>
<li>env.file</li>
<li>SQL query</li>
<li>acquire.py</li>
<li>acquire.ipynb</li>
<br>
<h4><b>Prepare</b></h4>
<li>Remove unwanted cols</li>
<li>Aggregate cols</li>
<li>Confirm veracity of cols</li>
<li>prepare.py</li>
<li>prepare.ipynb</li>
<br>
<h4><b>Explore</b></h4>
<li>Determine best cols</li>
<li>Do hypothesis null/alternate</li>
<li>Stat check best cols</li>
<li>explore.py</li>
<li>explore.ipynb</li>
<br>
<h4><b>Modeling</b></h4>
<li>Decision Tree</li>
<li>Best model</li>
<li>Random Forest</li>
<li>Best model</li>
<li>KNN</li>
<li>Best model</li>
<li>Linear Regression</li>
<li>Best model</li>
<li>TOP 3 COMPARISON</li>
<li>modeling.py</li>
<li>modeling.ipynb</li>
<br>
<h4><b>Delivery</b></h4>
<li>Final notebook (.ipynb)</li>
<li>Readme.md</li>
<br><br><br>






<!-- Instructions To Replicate -->
<head>
    <h3 align='center'><b><i>
        <a id='instructions'></a>Instructions To Replicate:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>

1. Clone this repo
2. Create 'env.py' file that connects to SQL
3. Run desired .ipynb
<br><br><br>





<!-- Takeaways -->
<head>
    <h3 align='center'><b><i>
        <a id='takeaways'></a>Takeaways:
    </i></b></h3>
</head>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>Key Findings:</b></h4>

- Less Likely to Churn
    - 'partner' = 'Yes'
    - 'dependents' = 'Yes'
    - 'online_security' = 'Yes', 'No internet service'
    - 'online_backup = 'No internet service'
    - 'device_protection' = 'Yes', 'No internet service'
    - 'tech_support' = 'Yes', 'No internet service'
    - 'streaming_tv' = 'No internet service'
    - 'streaming_movies' = 'No internet service'
    - 'contract_type' = 'One year', 'Two year'
    - 'total_services' = '1', '6', '7', '8', '9'
    - 'total_extra_services' = '0', '3', '4', '5', '6'
- AVOID
    - 'payment_type' = 'Electronic check'
    - 'contract_type' = 'Month-to-month'
    - 'total_services'= '3', '4'
    - 'total_extra_services' = '1', '2'
<br><br>
<h4><b>Recommendations:</b></h4>

- Focus advertisement efforts towards families and couples and avoid older population
- Market offers as single services or bundled services with at least 3 additional services
- For customers, establish whether or not they desire additional internet services
    - If yes, push bundles of at least 3 additional services
    - If no, do not allow additional services between 1 and 2
- For paying customers:
    - Do not allow monthly contracts
    - Do not allow electronic check forms of payment
<br><br>
<h4><b>Takeaways:</b></h4>

- Target audience:
    - Couples
    - Families/Dependents
- Services Emphasis:
    - Cell Services:
        - Not much difference with/without/multiple
    - Internet Services:
        - Either no internet or at least bundle 3 subs
- Payments:
    - DO NOT ALLOW ELECTRONIC CHECK
- Contracts:
    - DO NOT ALLOW MONTH CONTRACT
<br><br><br>