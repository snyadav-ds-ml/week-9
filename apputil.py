import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate="mean"):
        if estimate in ["mean","median"]:
            self.estimate = estimate
        else:
            raise ValueError(" only mean and median allowed")


    
    def fit(self, X, y,default_category=None):
        
        if len(X) != len(y):
            raise ValueError("Length of X and y should be same")
        elif pd.isnull(y).any():
            raise ValueError("y can not be null")
        df = X.copy()
        self.default_category = default_category
        df['label']=y
        self.columns =  list(X.columns)
        if default_category is not None :
            if default_category not in self.columns:
                raise ValueError("Defaul category not in DF")
            else:
                self.value = df.groupby(self.default_category, observed=True)["label"]
                
        else :
            self.value = df.groupby(self.columns)["label"]
        if self.estimate == "mean":
            self.value = self.value.mean()
        else:
            self.value = self.value.median()    

    def predict(self, X):
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        merged = pd.merge(
            X,
            self.value.reset_index(),
            how='left',
            on=self.default_category
        )

        
        # Count how many rows are missing (NaN predictions)
        missing_count = merged['label'].isnull().sum()
        if missing_count > 0 and self.default_category is None:
            print(f"Warning: {missing_count} observation(s) contain unseen groups. Returning NaN for them.")
        
        return merged['label'].to_numpy()