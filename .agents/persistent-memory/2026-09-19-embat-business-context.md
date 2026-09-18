# Embat — contexto de negocio para ideación de producto

- **Fecha de investigación:** 2026-09-19
- **Objetivo:** servir como contexto persistente para evaluar ideas construidas sobre el Health Score del reto X Ray.
- **Fuentes:** páginas oficiales de Embat, documentación de producto, changelog y casos de éxito. Los hechos llevan enlace; las interpretaciones están marcadas como tales.

## Resumen ejecutivo

Embat es un TMS cloud para equipos financieros de empresas medianas y grandes que han superado la gestión de tesorería en Excel. Conecta bancos, ERP, PSP y plataformas de gasto para centralizar caja, previsiones, deuda, operaciones intercompany, conciliación, contabilidad y pagos. Su tesis es convertir la tesorería de una tarea manual y retrospectiva en un sistema operativo en tiempo real para tomar decisiones.

Su producto ya se está moviendo desde la **visibilidad** hacia la **inteligencia y ejecución asistida** mediante TellMe, un agente de IA nativo. Por tanto, una idea fuerte para el hackathon debe:

1. aprovechar datos que Embat ya tiene;
2. integrarse en un flujo de trabajo existente;
3. entregar una decisión o acción, no otro dashboard;
4. respetar permisos, trazabilidad, reversibilidad y supervisión humana;
5. demostrar una capacidad diferencial frente a TellMe y Risk Management.

## Qué es Embat

Embat se define como una plataforma de gestión de tesorería corporativa creada “por expertos financieros para expertos financieros”. Sus fundadores combinan experiencia en banca de inversión y tecnología financiera; su misión declarada es liberar a los equipos de finanzas de tareas manuales y convertir la tesorería en un eje estratégico. Trabaja con empresas medianas y grandes y cuenta con oficinas en Madrid, Londres, Berlín y Múnich. [About Embat](https://www.embat.io/about-us)

Embat no sustituye al ERP. Su posición es ser el **sistema de acción para caja, bancos y liquidez**, mientras el ERP permanece como sistema contable de registro. La sincronización es bidireccional. [Treasury Management](https://www.embat.io/treasury-management)

## Qué ofrece

La plataforma se comercializa de forma modular y con pricing personalizado. [Pricing](https://www.embat.io/pricing)

| Área | Capacidades principales |
| --- | --- |
| Connectivity Hub | Conectividad bancaria, ERP, PSP y plataformas de gastos; APIs y protocolos como SWIFT, EBICS y host-to-host. |
| Cashflow Management & Forecasting | Posición de caja en tiempo real, categorización, previsiones, escenarios, análisis de desviaciones y alertas. |
| Intercompany | Seguimiento de deuda e intereses entre entidades, matching, liquidaciones y asientos automáticos. |
| Risk Management | Visibilidad de contrapartes y exposición de crédito, gestión de deuda, detección de riesgo con IA y controles de límites/condiciones. |
| Cash Accounting & Reconciliation | Conciliación bancaria y de PSP, contabilización y matching de transacciones/facturas. |
| Corporate Payments | Preparación, aprobación, firma, ejecución y reconciliación de pagos domésticos e internacionales. |
| TellMe | Agente de IA transversal para tesorería, contabilidad, pagos y optimización de caja. |

### Conectividad y superficie de datos

- Más de 15.000 instituciones financieras conectadas mediante APIs y canales de fichero.
- Integración bidireccional con SAP, Oracle NetSuite, Microsoft Dynamics/Business Central, Sage, DATEV y otros ERP.
- Integración con PSP como Stripe, PayPal, Adyen y otros, y plataformas de gastos como Pleo, Spendesk, Moss y Payhawk.
- La API pública permite leer y escribir transacciones bancarias, pagos, facturas, asientos, saldos y entidades relacionadas, con scope por `companyId` y credenciales de sandbox. [Financial integrations](https://www.embat.io/financial-integrations), [Embat API](https://api.embat.io/docs)

**Implicación para el reto:** Embat ya posee el contexto transaccional necesario para que el Health Score sea un módulo nativo. Una propuesta que dependa de introducir datos manualmente o construir una nueva capa de integración desaprovecha su ventaja principal.

## TellMe: patrón de producto que debemos respetar

TellMe es el agente de IA de Embat. Opera sobre cuatro procesos: tesorería, contabilidad, optimización de caja y pagos. Sus capacidades incluyen categorización, enriquecimiento, conciliación, ajuste de previsiones según comportamiento de pago, detección de huecos, proyección de caja, narrativa de reporting, optimización de liquidez y asistencia en pagos. [TellMe](https://www.embat.io/es/inteligencia-artificial-finanzas)

Tiene tres modos:

- **Silent:** ejecuta tareas rutinarias de fondo cuando existe suficiente confianza. Todo queda registrado y puede revertirse.
- **Guided:** presenta contexto, razonamiento y una propuesta para que el usuario apruebe, modifique o rechace.
- **Ask:** responde consultas en lenguaje natural sobre los datos de tesorería respetando los permisos de cada usuario.

TellMe aprende de las decisiones del usuario y prioriza la supervisión humana. Embat afirma que las acciones son trazables, reversibles y restringidas por permisos. [TellMe Help Center](https://help.embat.io/hc/en-us/articles/34225505889949-TellMe-what-it-is-and-how-it-works), [Changelog](https://www.embat.io/changelog), [Security](https://www.embat.io/es/seguridad)

**Implicación para el reto:** la forma más creíble de integrar una idea agentic es como una nueva *skill* de TellMe, no como otro asistente paralelo. Debe decidir cuándo trabajar en Silent, cuándo elevar a Guided y qué puede responder en Ask.

## ICP: cliente ideal

### Segmentos declarados

- Empresas mid-market en crecimiento.
- Grupos corporativos multi-entidad.
- Operaciones multibanco y multidivisa.
- Empresas de rápido crecimiento o respaldadas por private equity.
- Multinacionales que incorporan entidades, países, bancos o adquisiciones con frecuencia.

Embat cita como señales de madurez haber superado aproximadamente cinco cuentas bancarias, operar con cinco o más entidades legales o entrar diariamente en más de diez portales bancarios. La web inglesa usa como ejemplo grupos por encima de £15M de facturación y la española menciona €25M; deben interpretarse como señales comerciales orientativas, no como una regla universal de cualificación. [Treasury Management](https://www.embat.io/treasury-management), [Gestión de tesorería](https://www.embat.io/es/gestion-tesoreria)

### Rasgos operativos del ICP

- La tesorería ya no cabe cómodamente en Excel.
- Varias sociedades, bancos, monedas, países o canales de venta.
- ERP consolidado pero desconectado del dato bancario en tiempo real.
- Mucha conciliación, clasificación y consolidación manual.
- Necesidad de control, auditoría y segregación de funciones.
- Crecimiento que añade complejidad más rápido que headcount financiero.
- Decisiones de liquidez y financiación con impacto material.

### Sectores observados

El producto es horizontal. Sus casos públicos incluyen retail, entretenimiento, construcción, educación, industria, tecnología, hospitality, movilidad, marketing, real estate y otros. La variable que une a los clientes no es el sector, sino la **complejidad financiera y operativa**. [Customer stories](https://www.embat.io/success-stories)

## Usuarios y buying committee

| Perfil | Rol en la compra/uso | Necesidad principal |
| --- | --- | --- |
| CFO | Comprador económico y sponsor | Liquidez, riesgo, financiación, capital allocation y reporting ejecutivo. |
| Finance Director / Head of Finance | Comprador y owner funcional | Escalabilidad, control, automatización y rendimiento del equipo. |
| Treasury Manager / Tesorero | Usuario principal y champion | Posición diaria, previsiones, bancos, deuda, divisas, pagos y alertas. |
| Corporate Finance | Usuario analítico | Deuda, cash pooling, financiación, intercompany y escenarios. |
| Controller / FP&A | Usuario analítico | Forecast, desviaciones, presupuesto y reporting. |
| Accounting / Reconciliation | Usuario operativo | Conciliación, contabilización, cierre y trazabilidad. |
| Accounts Payable | Usuario operativo | Preparación, aprobación y ejecución de pagos. |
| Accounts Receivable / Credit | Usuario operativo | Cobros, facturas vencidas, comportamiento de pago y exposición. |
| CEO, consejo, accionistas o PE sponsor | Consumidor de output | Resumen de liquidez, riesgo y capacidad de inversión. |
| IT, seguridad y auditoría | Gatekeeper | Integración, permisos, cumplimiento, fiabilidad y audit trail. |

**Implicación para el reto:** cada alerta o recomendación debe tener un dueño operativo y una escalación ejecutiva. “Para el CFO” es demasiado genérico si no se define quién trabaja realmente con la señal.

## Jobs to be done y dolores recurrentes

1. Saber cuánto efectivo existe ahora, consolidado por banco, entidad, país y moneda.
2. Entender qué pasará con la caja y por qué el forecast se desvía.
3. Detectar retrasos, exposiciones, covenants, brechas o inconsistencias antes del cierre.
4. Mover liquidez entre entidades y divisas con menor coste y riesgo.
5. Centralizar y controlar pagos sin saltar entre portales bancarios.
6. Conciliar y contabilizar grandes volúmenes con menos trabajo manual.
7. Incorporar nuevas entidades, bancos o adquisiciones sin rehacer la infraestructura.
8. Demostrar trazabilidad y control ante auditoría, compliance y consejo.
9. Liberar al equipo para análisis, negociación y decisiones estratégicas.

## Evidencia de clientes

| Cliente | Complejidad/dolor | Resultado público |
| --- | --- | --- |
| HOFF | Crecimiento de €5M a €65M, más bancos, monedas y transacciones | Reducción del 85–90% del tiempo operativo de tesorería; una persona dedica ahora 10–15% de su tiempo a la gestión diaria. [Caso HOFF](https://www.embat.io/success-stories/hoff) |
| Fever | Operación internacional y consolidación manual banco a banco | Pool bancario consolidado en minutos; infraestructura financiera para operar en más de 50 países. [Caso Fever](https://www.embat.io/success-stories/fever) |
| thePower | Pagos B2C, B2B y B2G en 60 países | Cierre adelantado más de cuatro días; tiempo por pago de 10 minutos a segundos. [Treasury Management](https://www.embat.io/treasury-management) |
| Molins | Forecast manual, visión estática, cash pooling y operación en 11 países | Visibilidad sobre actuals, previsión confirmada y estimada; automatización de forecast y cash pooling. [Caso Molins](https://www.embat.io/es/casos-de-exito/molins) |
| Wallapop | Alto volumen, conciliación manual y visibilidad incompleta | Tesorería en tiempo real y automatización de contabilidad/conciliación con NetSuite. [Caso Wallapop](https://www.embat.io/es/casos-de-exito/wallapop) |
| Northern Data | Más de 100 cuentas, pagos multidivisa y reporting de capital | Pagos de tres días a menos de medio día; 95% de contabilidad automatizada; forecast usado por consejo y accionistas. [Caso Northern Data](https://www.embat.io/success-stories/northern-data) |

## Posicionamiento y modelo comercial

- Plataforma B2B modular con propuesta y precio personalizados.
- Implantación orientada a semanas: Embat comunica 4–6 semanas para grupos mid-market y un trimestre para funcionalidad completa, frente a proyectos legacy mucho más largos.
- Venta basada en ROI operativo, visibilidad en tiempo real, escalabilidad y control.
- Expansión natural por módulos: conectividad/visibilidad → forecast → conciliación → pagos → riesgo/intercompany → IA.
- Ecosistema de partners tecnológicos y financieros como canal de distribución e integración. [Pricing](https://www.embat.io/pricing), [Partners](https://www.embat.io/es/partners)

## Seguridad y guardrails relevantes

- ISO 27001, SOC 2, GDPR y controles asociados a conectividad bancaria.
- Acceso por roles y principio de mínimo privilegio.
- Integraciones con entornos separados, cifrado y audit logs.
- IA con human-in-the-loop; los usuarios conservan control sobre operaciones financieras.
- Acciones y sugerencias explicables, trazables y reversibles.
- TellMe mantiene el scope de permisos por usuario y entidad. [Security](https://www.embat.io/security), [TellMe security](https://help.embat.io/hc/es/articles/34553832974877-Seguridad-y-privacidad-de-TellMe)

## Qué ya existe y no conviene duplicar

Según la web y el changelog, Embat ya ofrece o anuncia:

- desviaciones y alertas de forecast;
- análisis histórico del comportamiento de pago;
- ajuste de fechas de previsión;
- categorización y conciliación automática;
- detección de riesgo de contrapartes con IA;
- gestión de exposición de crédito y deuda;
- análisis conversacional sobre datos propios;
- narrativa automática de reporting;
- optimización de distribución de liquidez;
- controles y aprobaciones sobre pagos.

Una idea que solo diga “alertas de caja”, “chat con tus datos” o “scoring de contrapartes” puede parecer una repetición de su roadmap actual.

## Espacios diferenciales para el reto

Estas son **inferencias**, no funcionalidades confirmadas como ausentes:

1. **Health trajectory a nivel de empresa/grupo:** un score único que sintetice liquidez, cobros, pagos, deuda y trayectoria, en ambas direcciones.
2. **Lead time medido:** demostrar cuántos meses antes se detectó un deterioro o una mejora, no solo generar una alerta.
3. **Persistencia vs. bache:** clasificar si una señal es transitoria o estructural usando varios meses y varios pilares.
4. **Routing organizativo:** llevar la misma señal al tesorero, Cobros, Controller o CFO con una acción adaptada a su responsabilidad.
5. **Explicación causal del cambio de score:** “qué movió el score, desde cuándo y cuánto contribuyó”.
6. **Portfolio view para grupos y PE:** ordenar entidades o participadas por salud, tendencia y urgencia.
7. **Opportunity detection:** identificar mejoras explotables —capacidad de refinanciación, liberación de buffer o mejores condiciones—, no solo riesgos.
8. **Closed-loop outcome:** registrar si la recomendación fue aceptada y si mejoró el score o el resultado financiero después.

## Criterios para filtrar ideas del hackathon

Una idea fuerte debería poder responder “sí” a la mayoría:

1. ¿Usa el Health Score de forma central y visible?
2. ¿Aprovecha datos que Embat ya conecta?
3. ¿Tiene un usuario diario y un comprador económico definidos?
4. ¿Produce una decisión o acción concreta?
5. ¿Detecta mejora además de deterioro?
6. ¿Mide anticipación y distingue bache de tendencia?
7. ¿Explica la señal de forma audit-ready?
8. ¿Puede encajar como módulo o skill de TellMe?
9. ¿Evita duplicar una capacidad ya visible en la oferta actual?
10. ¿Se puede demostrar en cinco minutos con datos del reto?

## Tesis para el brainstorming

El mejor terreno no es “otra herramienta financiera”, sino una **capa nueva de decisión sobre el grafo de datos que Embat ya posee**. El producto debería convertir el Health Score en una intervención dentro de un workflow existente: priorizar, escalar, simular, aprobar, negociar o medir el resultado.

La narrativa comercial más creíble es B2B2B: Embat incorpora el producto como módulo premium o skill de TellMe; sus clientes lo usan dentro de sus flujos actuales. Esto reduce fricción de integración, distribución y confianza.

## Gaps y cautelas

- La web pública no permite confirmar pricing, attach rate por módulo, uso real por perfil ni prioridades internas de roadmap.
- Algunas cifras son mensajes comerciales y casos seleccionados, no benchmarks independientes.
- Los umbrales de ICP varían por página/mercado; no deben convertirse en criterios rígidos.
- “No aparece en la web” no prueba que una capacidad no exista internamente o esté en desarrollo.
- Antes de cerrar una idea conviene preguntarle al equipo de Embat: “¿Qué parte de esto ya hace TellMe o Risk Management, y dónde sigue habiendo fricción para el usuario?”

