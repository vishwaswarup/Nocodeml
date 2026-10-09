# Example datasets

Small, synthetic CSV files to try NoCodeML with. Both are made up (no real people or homes) and include a few missing
values on purpose, so the preprocessing step has something to do.

| File | Task | Predict | Rows | Notes |
|---|---|---|---|---|
| `customer_churn.csv` | classification | `churned` (1 = cancelled) | 1,200 | has an ID column (`customer_id`, which NoCodeML will suggest dropping) and a date column (`signup_date`) |
| `house_prices.csv` | regression | `price` | 1,000 | numeric and categorical (`city`) features |

How to use one: create a project at https://nocodeml.vercel.app, upload the file, choose the column to predict, then follow the
sections on the left. The free hosted version accepts files up to 10 MB.
