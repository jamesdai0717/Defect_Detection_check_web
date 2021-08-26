from radiohead import app
from flask import render_template, redirect, url_for,flash,request
from radiohead.models import EXPORT_MONGO, SELECT_DATA, UPDATE_DATA, UPDATE_DATA2, SELECT_OK_DATA, SELECT_NG_DATA, User, P_OK_DATA, P_NG_DATA, M_DATA
from radiohead.forms import RegisterForm, LoginForm
from radiohead import db
from flask_login import login_user, logout_user, login_required

@app.route("/")
@app.route("/home")
def home_page():
    return render_template('home.html')

@app.route('/check',methods = ['GET','POST'])
@login_required
def check_page():
    data = EXPORT_MONGO('result')
    id = ''
    if request.method == "POST":
        order = request.form["ZORDNO_1320"]
        real_result = request.form["REAL_RESULT"]
        id = request.form['id']
        if real_result == ' ':
            UPDATE_DATA2('result',data,order)
        else:
            UPDATE_DATA('result',data,real_result,order)
    products = SELECT_DATA(data,x=id)
    okproducts = SELECT_OK_DATA(data,x=id)
    ngproducts = SELECT_NG_DATA(data,x=id)
    return render_template('check.html', **locals())

@app.route('/status',methods = ['GET','POST'])
@login_required
def status_page():
    data = EXPORT_MONGO('status')
    id = ''
    if request.method == "POST":
        id = request.form['id']
    predict_ng = P_NG_DATA(data,x=id)
    predict_ok = P_OK_DATA(data,x=id)
    missing = M_DATA(data,x=id)
    return render_template('status.html', **locals())

@app.route('/register', methods = ['GET','POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email_address=form.email_address.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        return redirect(url_for('check_page'))
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'There was as error with creating a user:{err_msg}',category='danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('check_page'))
        else:
            flash('Username and password are not match! Please try again', category='danger')

    return render_template('login.html', form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))
