"""AgentDojo tools -> Hermes inventory axes (Paket 3, inventar.json).

Axes (same definitions as the Hermes tool inventory):
egress, irreversibel, liest_sensible_daten, taint_quelle; values ja | bedingt | nein,
``bedingt`` counts as ``ja`` (Hermes rule, Paket 3 b/e).

Each entry names the closest Hermes inventory tool (``analog``) and the reason.
Keys in the gateway table are ``agentdojo.<tool>`` so AgentDojo names never
collide with Hermes names that have special handling (``read_file``,
``search_files``).
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

PREFIX = "agentdojo."


class Axes(NamedTuple):
    egress: str
    irreversibel: str
    sensibel: str
    taint: str
    analog: str
    grund: str


J, B, N = "ja", "bedingt", "nein"

# Shared shapes
_SEND = (J, J, N, N)
_MSG_READ = (N, N, J, J)
_PROVIDER_LOOKUP = (J, N, N, J)

TOOL_MAP: Dict[str, Dict[str, Axes]] = {
    "slack": {
        "get_channels": Axes(N, N, B, B, "kanban_list", "Kanalliste des Arbeitsbereichs; Namen von Dritten angelegt"),
        "add_user_to_channel": Axes(J, B, N, N, "discord_admin", "gibt Dritten Lesezugriff auf Kanalverlauf"),
        "read_channel_messages": Axes(*_MSG_READ, "feishu_drive_list_comments", "Nachrichten Dritter (Injektionsvektor)"),
        "read_inbox": Axes(*_MSG_READ, "feishu_drive_list_comments", "Direktnachrichten Dritter"),
        "send_direct_message": Axes(*_SEND, "yb_send_dm", "Nachricht verlässt den Nutzer, nicht rückholbar"),
        "send_channel_message": Axes(*_SEND, "yb_send_dm", "Nachricht an Kanal, nicht rückholbar"),
        "get_users_in_channel": Axes(N, N, J, B, "yb_query_group_members", "Mitgliederliste (Personendaten)"),
        "invite_user_to_slack": Axes(*_SEND, "discord_admin", "Einladung an externe Mail-Adresse"),
        "remove_user_from_slack": Axes(N, J, N, N, "discord_admin", "Löschen eines Nutzers"),
        "get_webpage": Axes(*_PROVIDER_LOOKUP, "web_extract", "URL verlässt Host, Inhalt fremdkontrolliert"),
        "post_webpage": Axes(*_SEND, "browser_type/web POST", "veröffentlicht Inhalt unter fremder URL"),
    },
    "banking": {
        "get_iban": Axes(N, N, J, N, "who_is_this", "eigene Kontodaten"),
        "send_money": Axes(*_SEND, "yb_send_dm (Klasse 3)", "Überweisung: Daten + Geld verlassen, Kosten"),
        "schedule_transaction": Axes(*_SEND, "cronjob", "geplante Überweisung"),
        "update_scheduled_transaction": Axes(*_SEND, "cronjob", "ändert Empfänger/Betrag"),
        "get_balance": Axes(N, N, J, N, "ha_list_entities", "Kontostand"),
        "get_most_recent_transactions": Axes(N, N, J, J, "session_search", "Verwendungszwecke von Dritten"),
        "get_scheduled_transactions": Axes(N, N, J, B, "session_search", "Daueraufträge"),
        "read_file": Axes(N, N, J, B, "read_file", "Datei (z. B. Rechnung von Dritten)"),
        "get_user_info": Axes(N, N, J, N, "who_is_this", "Personendaten"),
        "update_password": Axes(N, J, N, N, "memory", "überschreibt Zugangsdaten"),
        "update_user_info": Axes(N, J, N, N, "memory", "überschreibt Personendaten"),
    },
    "workspace": {
        "send_email": Axes(*_SEND, "yb_send_dm", "Mail verlässt den Nutzer"),
        "delete_email": Axes(N, J, N, N, "write_file", "Löschen"),
        "get_unread_emails": Axes(*_MSG_READ, "feishu_drive_list_comments", "Mails Dritter"),
        "get_sent_emails": Axes(N, N, J, B, "session_search", "eigene gesendete Mails"),
        "get_received_emails": Axes(*_MSG_READ, "feishu_drive_list_comments", "Mails Dritter"),
        "get_draft_emails": Axes(N, N, J, B, "session_search", "Entwürfe"),
        "search_emails": Axes(*_MSG_READ, "session_search", "Mails Dritter"),
        "search_contacts_by_name": Axes(N, N, J, B, "who_is_this", "Kontakte"),
        "search_contacts_by_email": Axes(N, N, J, B, "who_is_this", "Kontakte"),
        "get_current_day": Axes(N, N, N, N, "clarify", "Datum"),
        "search_calendar_events": Axes(*_MSG_READ, "session_search", "Termine inkl. Einladungstexte Dritter"),
        "get_day_calendar_events": Axes(*_MSG_READ, "session_search", "Termine inkl. Einladungstexte Dritter"),
        "create_calendar_event": Axes(J, B, N, N, "cronjob", "Einladung an Teilnehmer"),
        "cancel_calendar_event": Axes(B, J, N, N, "cronjob", "Absage an Teilnehmer"),
        "reschedule_calendar_event": Axes(B, B, N, N, "cronjob", "Änderung an Teilnehmer"),
        "add_calendar_event_participants": Axes(J, B, N, N, "cronjob", "Einladung an Dritte"),
        "append_to_file": Axes(N, J, N, N, "patch", "ändert Datei"),
        "search_files_by_filename": Axes(N, N, J, B, "search_files", "Dateien, z. T. von Dritten geteilt"),
        "create_file": Axes(N, J, N, N, "write_file", "legt Datei an"),
        "delete_file": Axes(N, J, N, N, "write_file", "Löschen"),
        "get_file_by_id": Axes(N, N, J, B, "read_file", "Datei, z. T. von Dritten"),
        "list_files": Axes(N, N, J, B, "search_files", "Dateiliste inkl. Inhalte"),
        "share_file": Axes(*_SEND, "drive_preview", "gibt Datei an Dritte"),
        "search_files": Axes(N, N, J, B, "search_files", "Dateiinhalte"),
    },
    "travel": {
        "get_user_information": Axes(N, N, J, N, "who_is_this", "Personendaten inkl. Ausweisnummer"),
        **{
            name: Axes(*_PROVIDER_LOOKUP, "web_search", "Anfrage an Anbieter, Inhalt fremdkontrolliert")
            for name in (
                "get_all_hotels_in_city", "get_hotels_prices", "get_rating_reviews_for_hotels",
                "get_hotels_address", "get_all_restaurants_in_city", "get_cuisine_type_for_restaurants",
                "get_restaurants_address", "get_rating_reviews_for_restaurants",
                "get_dietary_restrictions_for_all_restaurants", "get_contact_information_for_restaurants",
                "get_price_for_restaurants", "check_restaurant_opening_hours",
                "get_all_car_rental_companies_in_city", "get_car_types_available",
                "get_rating_reviews_for_car_rental", "get_car_fuel_options", "get_car_rental_address",
                "get_car_price_per_day", "get_flight_information",
            )
        },
        "create_calendar_event": Axes(J, B, N, N, "cronjob", "Einladung an Teilnehmer"),
        "search_calendar_events": Axes(*_MSG_READ, "session_search", "Termine"),
        "get_day_calendar_events": Axes(*_MSG_READ, "session_search", "Termine"),
        "cancel_calendar_event": Axes(B, J, N, N, "cronjob", "Absage"),
        "reserve_hotel": Axes(*_SEND, "yb_send_dm (Klasse 3)", "Buchung: Personendaten an Anbieter, Kosten"),
        "reserve_car_rental": Axes(*_SEND, "yb_send_dm (Klasse 3)", "Buchung"),
        "reserve_restaurant": Axes(*_SEND, "yb_send_dm (Klasse 3)", "Buchung"),
        "send_email": Axes(*_SEND, "yb_send_dm", "Mail verlässt den Nutzer"),
    },
}


def key(tool_name: str) -> str:
    return f"{PREFIX}{tool_name}"


def hermes_axes_table() -> Dict[str, Tuple[str, str, str, str]]:
    """Table in the format of ``agent.session_gateway_axes.TOOL_AXES``.

    Tools with the same name in several suites must carry identical axes.
    """
    table: Dict[str, Tuple[str, str, str, str]] = {}
    for suite, tools in TOOL_MAP.items():
        for name, ax in tools.items():
            val = (ax.egress, ax.irreversibel, ax.sensibel, ax.taint)
            k = key(name)
            if k in table and table[k] != val:
                raise ValueError(f"conflicting axes for {name} ({suite}): {table[k]} vs {val}")
            table[k] = val
    return table


def yes(v: str) -> bool:
    return v in (J, B)


def classes(ax: Axes) -> Dict[str, bool]:
    return {
        "egress_oder_irreversibel": yes(ax.egress) or yes(ax.irreversibel),
        "sensibel": yes(ax.sensibel),
        "taint_quelle": yes(ax.taint),
    }
