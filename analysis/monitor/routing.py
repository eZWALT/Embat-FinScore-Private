"""Who gets an alert and what to do (plan step 5 wording, owners from the plan: tesorero / CFO / Cobros).

Owner ids match the contract: "treasurer" (tesorero), "cfo", "collections" (Cobros). The owner follows the item that moved
the most (the reason with the largest points), because that is where the fix lives. Actions are concrete and short, in Spanish
(tú form). They never say "ingresos en riesgo" ("revenue at risk") and never quote a probability.
"""
from __future__ import annotations

OWNER_LABELS = {"treasurer": "Tesorero", "cfo": "CFO", "collections": "Cobros"}

ITEM_OWNER = {
    "delay_paid": "treasurer", "ap_overdue30": "treasurer", "runway": "treasurer", "neg_liq": "treasurer", "neg_episodes": "treasurer",
    "out_vol": "treasurer", "delay_coll": "collections", "ar_overdue30": "collections", "credit_note": "collections",
    "ds_ratio": "cfo", "fc_ratio": "cfo", "ds_increase": "cfo", "fc_increase": "cfo", "months_observed": "cfo", "active_share": "treasurer",
    "cust_tail": "cfo", "guard": "treasurer",
}
DEFAULT_ACTION = "Revisa con la empresa los factores que más han movido la puntuación."
ITEM_ACTION = {
    "delay_paid": "Revisa el calendario de pagos: qué facturas de proveedores ya están vencidas, y paga o acuerda nuevas fechas empezando por las mayores.",
    "ap_overdue30": "Liquida o renegocia los pagos a proveedores con más de 30 días de retraso, empezando por los mayores, antes de que los proveedores endurezcan las condiciones.",
    "runway": "Rehaz la previsión de caja de los próximos 3 meses y decide qué mover (pagos, cobros, línea de crédito) para mantener la cobertura de caja.",
    "neg_liq": "La caja ha sido negativa: comprueba que el descubierto o la línea de crédito lo cubre y programa cobros frente a las próximas salidas grandes.",
    "neg_episodes": "La caja cae por debajo de cero con frecuencia: fija una regla de saldo mínimo y una fuente de financiación para esos huecos.",
    "out_vol": "Las salidas oscilan mucho: lista los pagos irregulares y prográmalos, para que el plan de caja pueda apoyarse en un nivel base.",
    "delay_coll": "Reclama los cobros vencidos, empezando por los más antiguos y de mayor importe, y revisa las condiciones de pago con los clientes más lentos.",
    "ar_overdue30": "Lanza una ronda de cobros sobre las facturas con más de 30 días de retraso y escala las de mayor importe.",
    "credit_note": "Revisa por qué se revierte tanta facturación con notas de crédito (disputas, errores, devoluciones) y corrige el origen.",
    "ds_ratio": "El pago de deuda se lleva más de cada euro cobrado: revisa el calendario de amortización y si refinanciar o reprogramar ayuda.",
    "fc_ratio": "Las comisiones e intereses bancarios se llevan una parte mayor: compara condiciones entre bancos y renegocia las más costosas.",
    "ds_increase": "El pago de deuda sube frente a las entradas de caja: comprueba qué deuda nueva lo provoca y si el plan de caja lo cubre.",
    "fc_increase": "Las comisiones e intereses suben frente a las entradas de caja: identifica qué cuenta o línea lo provoca y renegocia.",
    "months_observed": "Historial corto: trata la puntuación como provisional hasta que se conecten más meses.",
    "active_share": "Entra dinero en menos meses: comprueba si se ha dejado de facturar o si faltan cuentas en la conexión bancaria.",
    "cust_tail": "Un solo cliente concentra casi toda la facturación: acuerda condiciones y un límite de concentración, y empieza a diversificar la cartera.",
    "guard": "La actividad ha caído de golpe: confirma que las conexiones bancarias están completas y después contacta con la empresa.",
}
CATEGORY_OWNER = {"payment_history": "treasurer", "amounts_owed": "treasurer", "stability": "cfo", "new_credit": "cfo", "mix": "collections"}

DARK_ACTION = ("Comprueba que todas las cuentas bancarias siguen conectadas y enviando datos. Si la conexión está completa, llama hoy a la empresa: "
               "lleva 60 días o más sin movimientos bancarios.")
IMPROVE_ACTION = ("Confirma que la mejora es duradera (se ha mantenido en 3 de los últimos 4 meses). Si es así, es buen momento para revisar líneas de crédito "
                  "y condiciones de pago, o para dar uso a la caja ociosa.")
GROUP_ACTION = "Mira los miembros que más han caído (listados en la evidencia) y trata cada uno con la acción de su propia alerta."


def top_customer_action(customer: str, open_eur: float | None) -> str:
    from product.score.explain import eur

    base = f"Contacta con el cliente {customer}: pregunta si hay un pedido pendiente o si ha cambiado la relación."
    if open_eur and open_eur > 0:
        return base + f" Revisa los {eur(open_eur)} que siguen abiertos con él y empieza el cobro si están vencidos."
    return base + " Revisa cualquier exposición y las condiciones de crédito que le das."


def route_item(item: str | None) -> tuple[str, str]:
    """(owner, action) from the reason item that moved the most."""
    if item is None:
        return "cfo", DEFAULT_ACTION
    return ITEM_OWNER.get(item, "cfo"), ITEM_ACTION.get(item, DEFAULT_ACTION)
