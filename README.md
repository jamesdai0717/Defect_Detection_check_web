# defect_detection
* This project contains two parts. One is detect model; The Other is check web.
## detect_model
1. Consider interactions to let the value of certain variables compare in the same specifications.
2. Use the self-paced ensemble method to deal with the imbalanced data.
3. Choose the best model parameters and threshold at which the TPR is as large as possible conditional on TNR equals to 1 by training the model by both Balanced Random Forest and LightGBM.
4. Evaluating the model by testing data. (TNR=1,TPR=0.717,AUC=0.95)

## check_web
1. Collect the results in detect model and show the products that predicted NG in model since those products need to be confirm again.
2. Therefore, this web is to let user to be convenient to confirm the products and also renew directly the data in mongodb.
3. This web enable users to login, logout, register and search for their interesting product.

### Display
### Home Page
* Go to Login Page by clicking the Get started button!

![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/home_page.PNG?raw=true)
### Login Page
* Log in to go to Check Page!
* Click the Register button if you dont have the account!

![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/login_page.png?raw=true)
### Check Page
![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/check_page.PNG?raw=true)
### Confirm OK Page
![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/confirmok.png?raw=true)
### Confirm NG Page
![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/logout_page.png?raw=true)
### Search Page
![alt text](https://github.com/jamesdai0717/defect_detection/blob/main/check_web/images/search1.png?raw=true)
