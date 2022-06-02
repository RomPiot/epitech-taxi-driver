from django import forms


class QLearningParamsForm(forms.Form):
    nb_episodes = forms.IntegerField(help_text="default = 10 000", initial=10000)
    epsilon_rate = forms.FloatField(help_text="default = 1.0", initial=1.0)
    epsilon_min = forms.FloatField(help_text="default = 0.01", initial=0.01)
    epsilon_max = forms.FloatField(help_text="default = 1.0", initial=1.0)
    learning_rate = forms.FloatField(help_text="default = 0.8", initial=0.8)
    reward_discount_rate = forms.FloatField(help_text="default = 0.786", initial=0.786)
    decay_rate = forms.FloatField(help_text="default = 0.07", initial=0.07)
