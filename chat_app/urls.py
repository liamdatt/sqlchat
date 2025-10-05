from django.urls import path
from . import views

app_name = 'chat_app' # Add an app_name for namespacing if you have multiple apps
 
urlpatterns = [
    path('', views.chat_view, name='chat_view'),
    path('ask/', views.ask_agent_view, name='ask_agent'),
] 