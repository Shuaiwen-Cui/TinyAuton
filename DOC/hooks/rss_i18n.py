"""Keep RSS configuration reusable across mkdocs-static-i18n builds."""
from datetime import datetime

from mkdocs.plugins import event_priority


@event_priority(100)
def on_config(config):
    # RSS parses this setting in place. i18n runs on_config again for each
    # language with the same plugin instance; restore the documented input type.
    for plugin in config.plugins.values():
        if plugin.__class__.__module__.startswith("mkdocs_rss_plugin"):
            date_config = plugin.config.date_from_meta
            if isinstance(date_config.default_time, datetime):
                date_config.default_time = date_config.default_time.strftime("%H:%M")
    return config
