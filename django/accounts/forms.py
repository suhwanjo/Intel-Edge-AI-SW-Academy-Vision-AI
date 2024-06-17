from django import forms


class DateTimeForm(forms.Form):
    date = forms.DateField(widget=forms.SelectDateWidget)
    period = forms.ChoiceField(
        choices=[
            ('', '전체 (9:00-16:50)'),
            ('1', '1교시(9:00-9:50)'),
            ('2', '2교시(10:00-10:50)'),
            ('3', '3교시(11:00-11:50)'),
            ('4', '4교시(13:00-13:50)'),
            ('5', '5교시(14:00-14:50)'),
            ('6', '6교시(15:00-15:50)'),
            ('7', '7교시(16:00-16:50)'),
        ],
        required=False
)