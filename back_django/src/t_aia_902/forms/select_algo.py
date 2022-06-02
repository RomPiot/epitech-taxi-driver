from django import forms


class SelectAlgoForm(forms.Form):
    algo = forms.ChoiceField(
        choices=[
            ("q-learning", "Q-Learning"),
            ("deep-q-learning", "Deep Q-Learning"),
            ("sarsa", "Sarsa"),
        ]
    )
