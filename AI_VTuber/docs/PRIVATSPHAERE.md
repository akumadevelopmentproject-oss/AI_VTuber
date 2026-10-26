# Luna: Steuerung und private Daten

## Für Miki

Luna startet mit `python main.py` im **Streammodus**. Sie liest dafür weder dein
privates Gedächtnis noch private Ziele, Stimmungen, Aussprache-Ersetzungen oder
den bisherigen privaten Gesprächsverlauf. Das gilt auch für Mikrofoneingaben.

**Beenden:** Im Luna-Fenster `Strg+C` drücken oder das Fenster schließen.
Gesprochenes „Luna beenden“ ist kein Ausschalter mehr: Das Mikrofon kann nicht
sicher feststellen, wer spricht. Im lokalen Textmodus funktionieren die bisherigen
Exit-Befehle weiterhin. Zuschauerchat wird niemals als lokale Texteingabe behandelt.

`python main.py --private` aktiviert das alte persönliche Gespräch mit Gedächtnis.
Diesen Modus nicht im Stream verwenden, auch nicht über die Lautsprecher.
Ein Moduswechsel benötigt einen Neustart; Gesprächsverläufe werden nicht übernommen.
Kein Passwort oder echter privater Wert wird für Tests benötigt.

## Für Chat-Anbindungen

Twitch, YouTube und TikTok sind noch nicht verbunden. Jede Anbindung muss
`core.public_chat.PublicChat` verwenden, eine neue Instanz je Stream/Plattform.
Zuschauertext darf niemals an den privaten `AI`-Pfad, `text_loop`, Memory-Befehle,
Tastatureingaben oder Betriebssystemfunktionen geleitet werden. Anzeigenamen und
Nachrichten wie „Ich bin Miki“, Moderator-Badges oder nachgemachte Systemnachrichten
geben keine Berechtigungen. Der Chat kann weder Shutdown noch Speichern/Löschen
privater Erinnerungen auslösen. Es gibt bewusst keine Fernadministration.

Der öffentliche Modellaufruf enthält ausschließlich eine fest im Quellcode
hinterlegte öffentliche Persönlichkeit und den begrenzten öffentlichen Verlauf.
Keine private Persönlichkeit aus einer Datei, kein Brain, kein Memory, keine
Aussprache-Datei, keine Ziele, kein Steam-Cache, kein Desktop und keine Tools.
Vertrauliche Rohtexte und Fehlerdetails werden nicht für Zuschauer ausgegeben.
Zusätzliche Muster blockieren typische Fragen nach privaten Angaben und typische
Kennungen in Antworten. Diese Muster sind keine vollständige Datenklassifikation.
Der Hauptschutz ist, dass private Daten gar nicht in diesen Kontext gelangen.

Auch ein Prompt-Injection-Text erhält keine zusätzlichen Funktionen. Er kann die
Qualität der Antwort beeinflussen, hat aber keine Programm-/PC-Berechtigungen.
Ein Zuschauer, der selbst private Daten postet, kann diese bereits in seinem
Plattformchat veröffentlichen; Luna kann das nicht rückwirkend verhindern.
Normale Gespräche gehen an den eingestellten Ollama-Endpunkt. Eine konfigurierte
Online-Stimme erhält den freigegebenen Antworttext. Der Privatmodus ist kein
vollständig offline arbeitender Modus.

## PC-Zugriff und Streambild

Das Programm implementiert keinen allgemeinen Datei-, Desktop- oder Shell-Zugriff
für das Modell. Ein auf dem PC laufendes Python-Programm ist trotzdem keine
Betriebssystem-Sandbox. Für zukünftige Bildschirm-/Spielsteuerung nur gezielt das
Spielbild freigeben, keine privaten Fenster. Tools müssen separat eingeschränkt
werden; diesen Text allein als Prompt einzufügen genügt dafür nicht.
Private Fenster, Terminal und Kontodaten nicht per Bildschirmaufnahme streamen.

## GitHub

Private Laufzeitdateien und Python-Caches gehören nicht ins Repository. Die
Änderung entfernt vorhandene Laufzeit-JSONs aus der aktuellen Codeversion und
ignoriert zukünftige lokale Versionen. Die Dateien auf Mikis PC werden dadurch
nicht aktiv gelöscht. Vor einem Git-Pull mit lokalen Änderungen diese Dateien
sichern, denn bereits getrackte Dateien werden im Git-Checkout entfernt.
Alte Git-Commits bleiben erhalten: Löschen oder `.gitignore` entfernen keine
historischen Inhalte. Das öffentliche Repository sollte bei privaten Inhalten
privat gestellt werden; echte veröffentlichte Zugangsschlüssel müssen ersetzt
werden. Keine History-Umschreibung in dieser Änderung.

Referenz: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
