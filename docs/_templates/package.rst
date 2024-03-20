
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

    {% block modules %}
    {% if modules %}
    .. autosummary::
        :toctree:
        :recursive:
        {% for item in modules %}
            {{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}

    {% if classes %}
    .. rubric:: Classes

    .. autosummary::
        :toctree: .
        {% for class in classes %}
            {{ class }}
        {% endfor %}

    {% endif %}

    {% if functions %}
    .. rubric:: Functions

    .. autosummary::
        :toctree: .
        {% for function in functions %}
        {{ function }}
        {% endfor %}

    {% endif %}
