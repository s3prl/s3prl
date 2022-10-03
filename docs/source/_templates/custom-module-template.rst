{{ name | escape | underline}}

**({{ fullname }})**

.. automodule:: {{ fullname }}

{% block modules %}
{% if modules %}

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block attributes %}
{% if attributes %}
{% for item in attributes %}

{{ item }}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: {{ item }}

{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
{% for item in classes %}

{{ item }}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: {{ item }}
   :member-order: bysource
   :members:
   :undoc-members:
   :inherited-members: torch.nn.Module,nn.Module,Module
   :show-inheritance:

{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
{% for item in functions %}

{{ item }}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: {{ item }}

{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
{% for item in exceptions %}

{{ item }}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: {{ item }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

{%- endfor %}
{% endif %}
{% endblock %}
