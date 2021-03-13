# from django.urls import path
from django.conf.urls import url
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
# router.register(r'annotation_data', views.AnnotationDataViewSet)

urlpatterns = [
    url('project_info/', views.project_info, name='project_info'),
    url('upload_remote_file/', views.upload_remote_file, name='upload_remote_file'),
    url('load_local_dataset/', views.load_local_dataset, name='load_local_dataset'),
    url('export_data/', views.export_data, name='export_data'),
    url('load_single_unlabeled/', views.load_single_unlabeled, name='load_single_unlabeled'),
    url('annotate_single_unlabeled/', views.annotate_single_unlabeled, name='annotate_single_unlabeled'),
    url('query_annotatoin_history/', views.query_annotatoin_history, name='query_annotatoin_history'),

]
